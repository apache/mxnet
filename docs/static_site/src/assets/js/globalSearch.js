docsearch({
    apiKey: '500f8e78748bd043cc6e4ac130e8c0e7',
    indexName: 'apache_mxnet',
    inputSelector: '#global-search',
    algoliaOptions: { 'facetFilters': ["version:master"] },
    debug: true// Set debug to true if you want to inspect the dropdown
});