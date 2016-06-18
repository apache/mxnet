function changeLanguage(langSelect, langSelectLabel, rootpath){
	langSelect.change(function() {
		var lang = langSelect.val();
		if(lang == 'zh'){
			location.href = rootpath + 'zh/index.html';
		} else {
			location.href = rootpath + 'index.html';	
		}
	});
}

$(document).ready(function () {
	var langSelect = $("#lang-select");
	var langSelectLabel = $("#lang-select-label > span");
	currHref = location.href;
	
	if(/\/zh\//.test(currHref)){
		langSelect.val("zh");
	} else {
		langSelect.val("en");
	}
	langSelectLabel.text($("option:selected").text());

	changeLanguage(langSelect, langSelectLabel, getRootPath());
})