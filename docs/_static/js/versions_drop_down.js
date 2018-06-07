$(document).ready(function () {
      function loadVersionURL() {
          var el = $(this);
          console.log("loadVersionURL");
          window.location.pathname = '/versions/' + $(this).text().substr(1) + window.location.pathname ;
      }
      $('.opt-group').on('click', '.versions', loadVersionURL);
});
