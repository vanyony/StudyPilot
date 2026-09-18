(function () {
  "use strict";

  // The server remains the source of truth.  This only prevents an accidental
  // second click while the current form request is in flight.
  document.querySelectorAll("[data-answer-form]").forEach(function (form) {
    form.addEventListener("submit", function (event) {
      if (form.dataset.submitted === "true") {
        event.preventDefault();
        return;
      }
      var message = form.querySelector('input[name="message_id"]');
      if (message && !message.value) {
        message.value = (window.crypto && window.crypto.randomUUID)
          ? window.crypto.randomUUID()
          : "answer-" + Date.now() + "-" + Math.random().toString(16).slice(2);
      }
      form.dataset.submitted = "true";
      var submit = form.querySelector('button[type="submit"]');
      if (submit) {
        submit.disabled = true;
        submit.textContent = "提交中…";
      }
    });
  });
})();
