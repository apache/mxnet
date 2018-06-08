$(document).ready(function () {
      function loadVersionURL() {
          let el = $(this);
          console.log("loadVersionURL");
          let versionString = '';
          if ($(this).text().includes("master")) {
              versionString = $(this).text();
          } else {
              //Remove the character v at the beginning
              versionString = $(this).text().substr(1);
          }
          window.location.pathname = '/versions/' + versionString + window.location.pathname ;
      }
      $('.opt-group').on('click', '.versions', loadVersionURL);
});

