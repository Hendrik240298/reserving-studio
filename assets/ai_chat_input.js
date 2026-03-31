(function () {
  function wireChatEnterToSend() {
    var textarea = document.getElementById('ai-chat-input');
    var button = document.getElementById('ai-chat-send');
    if (!textarea || !button || textarea.dataset.enterWired === 'true') {
      return;
    }
    textarea.dataset.enterWired = 'true';
    textarea.addEventListener('keydown', function (event) {
      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        button.click();
      }
    });
  }

  function wireTranscriptAutoScroll() {
    var transcript = document.getElementById('ai-chat-transcript');
    if (!transcript || transcript.dataset.scrollWired === 'true') {
      return;
    }
    transcript.dataset.scrollWired = 'true';

    var scrollToBottom = function () {
      window.requestAnimationFrame(function () {
        transcript.scrollTop = transcript.scrollHeight;
      });
    };

    var observer = new MutationObserver(scrollToBottom);
    observer.observe(transcript, {
      childList: true,
      subtree: true,
      characterData: true,
    });

    scrollToBottom();
  }

  function wireChatUi() {
    wireChatEnterToSend();
    wireTranscriptAutoScroll();
  }

  document.addEventListener('DOMContentLoaded', wireChatUi);
  window.setInterval(wireChatUi, 800);
})();
