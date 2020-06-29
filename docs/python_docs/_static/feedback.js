$(document).ready(function() {
  $(".feedback-answer").on("click", function () {
    $(".feedback-question").remove();
    $(".feedback-answer-container").remove();
    $(".feedback-thank-you").show();
    ga("send", {
      hitType: "event",
      eventCategory: "Did this page help you?",
      eventAction: $(this).attr("data-response"),
      eventLabel: window.location.pathname || "unknown",
    });
  });
});
