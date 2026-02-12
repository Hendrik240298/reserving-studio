/* global window, document, localStorage, crypto */
(function () {
  var namespace = "__reserving_tab_sync__";
  if (window[namespace]) {
    return;
  }

  var state = {
    tabId: null,
    userKey: null,
    channelName: null,
    channel: null,
    storageKey: null,
    listenersInitialized: false,
    storageListenerBound: false,
  };

  var STORAGE_PREFIX = "reserving-sync-storage:";

  function buildTabId() {
    if (window.crypto && typeof window.crypto.randomUUID === "function") {
      return window.crypto.randomUUID();
    }
    return "tab-" + Math.random().toString(16).slice(2) + "-" + Date.now();
  }

  function ensureTabId() {
    if (!state.tabId) {
      state.tabId = buildTabId();
    }
    return state.tabId;
  }

  function emitToDash(payload) {
    var input = document.getElementById("sync-inbox");
    if (!input) {
      return;
    }
    var nextValue = JSON.stringify(payload);
    var valueSetter = Object.getOwnPropertyDescriptor(
      window.HTMLInputElement.prototype,
      "value"
    );
    if (valueSetter && typeof valueSetter.set === "function") {
      valueSetter.set.call(input, nextValue);
    } else {
      input.value = nextValue;
    }
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  }

  function handlePayload(payload) {
    if (!payload || payload.type !== "session_changed") {
      return;
    }
    if (!state.userKey && payload.user_key) {
      state.userKey = String(payload.user_key);
      state.storageKey = STORAGE_PREFIX + state.userKey;
    }
    if (!state.userKey || payload.user_key !== state.userKey) {
      return;
    }
    if (!payload.origin_tab_id || payload.origin_tab_id === state.tabId) {
      return;
    }
    emitToDash(payload);
  }

  function ensureListeners() {
    if (!state.storageListenerBound) {
      window.addEventListener("storage", function (event) {
        if (!event || !event.key || !event.newValue) {
          return;
        }
        var matchesUserScope = state.storageKey
          ? event.key === state.storageKey
          : event.key.indexOf(STORAGE_PREFIX) === 0;
        if (!matchesUserScope) {
          return;
        }
        try {
          var payload = JSON.parse(event.newValue);
          handlePayload(payload);
        } catch (_error) {
          return;
        }
      });
      state.storageListenerBound = true;
    }

    if (state.listenersInitialized) {
      return;
    }

    if (!state.channelName) {
      return;
    }

    if (typeof window.BroadcastChannel === "function" && state.channelName) {
      state.channel = new window.BroadcastChannel(state.channelName);
      state.channel.onmessage = function (event) {
        handlePayload(event && event.data);
      };
    }

    state.listenersInitialized = true;
  }

  function configure(userKey, tabId) {
    var nextUserKey = String(userKey || "default");
    var nextTabId = String(tabId || ensureTabId());

    if (
      state.userKey === nextUserKey &&
      state.tabId === nextTabId &&
      state.listenersInitialized
    ) {
      return nextTabId;
    }

    state.userKey = nextUserKey;
    state.tabId = nextTabId;
    state.channelName = "reserving-sync:" + state.userKey;
    state.storageKey = "reserving-sync-storage:" + state.userKey;

    if (state.channel && typeof state.channel.close === "function") {
      state.channel.close();
    }
    state.channel = null;
    state.listenersInitialized = false;

    ensureListeners();
    return state.tabId;
  }

  function publish(message) {
    if (!message || !state.userKey) {
      return;
    }
    var payload = {
      type: "session_changed",
      user_key: state.userKey,
      sync_version: Number(message.sync_version || 0),
      origin_tab_id: state.tabId,
      updated_at: message.updated_at || new Date().toISOString(),
    };

    if (state.channel) {
      state.channel.postMessage(payload);
    }

    try {
      localStorage.setItem(state.storageKey, JSON.stringify(payload));
      localStorage.removeItem(state.storageKey);
    } catch (_error) {
      return;
    }
  }

  window.ReservingTabSync = {
    getOrCreateTabId: ensureTabId,
    configure: configure,
    publish: publish,
  };

  window[namespace] = true;
  ensureListeners();
})();
