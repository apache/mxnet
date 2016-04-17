function auto_index(module) {
  $(document).ready(function () {
    // find all classes or functions
    var targets = $("dl[class='class'] > dt,dl[class='function'] > dt");
    for (var i = 0; i < targets.length; ++i) {
      console.log($(targets[i]).attr('id'));
    }

    var li_node = $("li a[href='#module-" + module + "']").parent();
    var html = "<ul>";

    for (var i = 0; i < targets.length; ++i) {
      var id = $(targets[i]).attr('id');
      id = id.replace(/^mxnet\./, ''); // remove 'mxnet.' prefix to make menus shorter
      html += "<li><a class='reference internal' href='#";
      html += id;
      html += "'>" + id + "</a></li>";
    }

    html += "</ul>";
    li_node.append(html);
    console.log(module);
  });
}

