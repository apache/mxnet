%module  "AI::NNVMCAPI"
%include typemaps.i
%rename("%(strip:[NN])s") "";
%init %{
    /* These SWIG_TypeClientData() calls might break in the future, but
     * %rename should work on these types before that happens. */
    SWIG_TypeClientData(SWIGTYPE_p_NNOp, (void *)"OpHandle");
    SWIG_TypeClientData(SWIGTYPE_p_NNSymbol, (void *)"SymbolHandle");
    SWIG_TypeClientData(SWIGTYPE_p_NNGraph, (void *)"GraphHandle");
%}
%inline %{
#include <c_api.h>
%} 
%include nnvm_typemaps.i

/*! \brief manually define unsigned int */
typedef unsigned int nn_uint;
/*! \brief handle to a function that takes param and creates symbol */
typedef NNOp *OpHandle;
/*! \brief handle to a symbol that can be bind as operator */
typedef NNSymbol *SymbolHandle;
/*! \brief handle to Graph */
typedef NNGraph *GraphHandle;
/*!
 * \brief Set the last error message needed by C API
 * \param msg The error message to set.
 */
void NNAPISetLastError(const char* msg);
/*!
 * \brief return str message of the last error
 *  all function in this file will return 0 when success
 *  and -1 when an error occured,
 *  NNGetLastError can be called to retrieve the error
 *
 *  this function is threadsafe and can be called by different thread
 *  \return error info
 */
const char *NNGetLastError(void);
/*!
 * \brief list all the available operator names, include entries.
 * \param out_size the size of returned array
 * \param out_array the output operator name array.
 * \return 0 when success, -1 when failure happens
 */
int NNListAllOpNames(nn_uint *out_size, const char*** out_array);
/*!
 * \brief Get operator handle given name.
 * \param op_name The name of the operator.
 * \param op_out The returnning op handle.
 */
int NNGetOpHandle(const char* op_name,
                           OpHandle* out);
/*!
 * \brief list all the available operators.
 *  This won't include the alias, use ListAllNames
 *  instead to get all alias names.
 *
 * \param out_size the size of returned array
 * \param out_array the output AtomicSymbolCreator array
 * \return 0 when success, -1 when failure happens
 */
int NNListUniqueOps(nn_uint *out_size,
                             OpHandle **out_array);
/*!
 * \brief Get the detailed information about atomic symbol.
 * \param op The operator handle.
 * \param real_name The returned name of the creator.
 *   This name is not the alias name of the atomic symbol.
 * \param description The returned description of the symbol.
 * \param num_doc_args Number of arguments that contain documents.
 * \param arg_names Name of the arguments of doc args
 * \param arg_type_infos Type informations about the arguments.
 * \param arg_descriptions Description information about the arguments.
 * \param return_type Return type of the function, if any.
 * \return 0 when success, -1 when failure happens
 */
int NNGetOpInfo(OpHandle op,
                         const char **real_name,
                         const char **description,
                         nn_uint *num_doc_args,
                         const char ***arg_names,
                         const char ***arg_type_infos,
                         const char ***arg_descriptions,
                         const char **return_type);
/*!
 * \brief Create an AtomicSymbol functor.
 * \param op The operator handle
 * \param num_param the number of parameters
 * \param keys the keys to the params
 * \param vals the vals of the params
 * \param out pointer to the created symbol handle
 * \return 0 when success, -1 when failure happens
 */
int NNSymbolCreateAtomicSymbol(OpHandle op,
                                        nn_uint num_param,
                                        const char **keys,
                                        const char **vals,
                                        SymbolHandle *out);
/*!
 * \brief Create a Variable Symbol.
 * \param name name of the variable
 * \param out pointer to the created symbol handle
 * \return 0 when success, -1 when failure happens
 */
int NNSymbolCreateVariable(const char *name, SymbolHandle *out);
/*!
 * \brief Create a Symbol by grouping list of symbols together
 * \param num_symbols number of symbols to be grouped
 * \param symbols array of symbol handles
 * \param out pointer to the created symbol handle
 * \return 0 when success, -1 when failure happens
 */
int NNSymbolCreateGroup(nn_uint num_symbols,
                                 SymbolHandle *in,
                                 SymbolHandle *out);
/*!
 * \brief Add src_dep to the handle as control dep.
 * \param handle The symbol to add dependency edges on.
 * \param src_dep the source handles.
 */
int NNAddControlDeps(SymbolHandle in,
                              SymbolHandle in);
/*!
 * \brief Free the symbol handle.
 * \param symbol the symbol
 * \return 0 when success, -1 when failure happens
 */
int NNSymbolFree(SymbolHandle in);
/*!
 * \brief Copy the symbol to another handle
 * \param symbol the source symbol
 * \param out used to hold the result of copy
 * \return 0 when success, -1 when failure happens
 */
int NNSymbolCopy(SymbolHandle in, SymbolHandle *out);
/*!
 * \brief Print the content of symbol, used for debug.
 * \param symbol the symbol
 * \param out_str pointer to hold the output string of the printing.
 * \return 0 when success, -1 when failure happens
 */
int NNSymbolPrint(SymbolHandle in, const char **out);
/*!
 * \brief Get string attribute from symbol
 * \param symbol the source symbol
 * \param key The key of the symbol.
 * \param out The result attribute, can be NULL if the attribute do not exist.
 * \param success Whether the result is contained in out.
 * \return 0 when success, -1 when failure happens
 */
int NNSymbolGetAttr(SymbolHandle in,
                             const char* key,
                             const char** out,
                             int *out);
/*!
 * \brief Set string attribute from symbol.
 *  NOTE: Setting attribute to a symbol can affect the semantics(mutable/immutable) of symbolic graph.
 *
 *  Safe recommendaton: use  immutable graph
 *  - Only allow set attributes during creation of new symbol as optional parameter
 *
 *  Mutable graph (be careful about the semantics):
 *  - Allow set attr at any point.
 *  - Mutating an attribute of some common node of two graphs can cause confusion from user.
 *
 * \param symbol the source symbol
 * \param num_param Number of parameters to set.
 * \param keys The keys of the attribute
 * \param values The value to be set
 * \return 0 when success, -1 when failure happens
 */
int NNSymbolSetAttrs(SymbolHandle in,
                              nn_uint num_param,
                              const char** keys,
                              const char** vals);
/*!
 * \brief Get all attributes from symbol, including all descendents.
 * \param symbol the source symbol
 * \param recursive_option 0 for recursive, 1 for shallow.
 * \param out_size The number of output attributes
 * \param out 2*out_size strings representing key value pairs.
 * \return 0 when success, -1 when failure happens
 */
int NNSymbolListAttrs(SymbolHandle in,
                               int recursive_option,
                               nn_uint *half_of_out_size,
                               const char*** out_array);
/*!
 * \brief List inputs variables in the symbol.
 * \param symbol the symbol
 * \param option The option to list the inputs
 *   option=0 means list all arguments.
 *   option=1 means list arguments that are readed only by the graph.
 *   option=2 means list arguments that are mutated by the graph.
 * \param out_size output size
 * \param out_sym_array the output array.
 * \return 0 when success, -1 when failure happens
 */
int NNSymbolListInputVariables(SymbolHandle in,
                                        int option,
                                        nn_uint *out_size,
                                        SymbolHandle** out_array);
/*!
 * \brief List input names in the symbol.
 * \param symbol the symbol
 * \param option The option to list the inputs
 *   option=0 means list all arguments.
 *   option=1 means list arguments that are readed only by the graph.
 *   option=2 means list arguments that are mutated by the graph.
 * \param out_size output size
 * \param out_str_array pointer to hold the output string array
 * \return 0 when success, -1 when failure happens
 */
int NNSymbolListInputNames(SymbolHandle in,
                                    int option,
                                    nn_uint *out_size,
                                    const char ***out_array);
/*!
 * \brief List returns names in the symbol.
 * \param symbol the symbol
 * \param out_size output size
 * \param out_str_array pointer to hold the output string array
 * \return 0 when success, -1 when failure happens
 */
int NNSymbolListOutputNames(SymbolHandle in,
                                     nn_uint *out_size,
                                     const char ***out_array);
/*!
 * \brief Get a symbol that contains all the internals.
 * \param symbol The symbol
 * \param out The output symbol whose outputs are all the internals.
 * \return 0 when success, -1 when failure happens
 */
int NNSymbolGetInternals(SymbolHandle in,
                                  SymbolHandle *out);
/*!
 * \brief Get index-th outputs of the symbol.
 * \param symbol The symbol
 * \param index the Index of the output.
 * \param out The output symbol whose outputs are the index-th symbol.
 * \return 0 when success, -1 when failure happens
 */
int NNSymbolGetOutput(SymbolHandle in,
                               nn_uint index,
                               SymbolHandle *out);

/*!
 * \brief Compose the symbol on other symbols.
 *
 *  This function will change the sym hanlde.
 *  To achieve function apply behavior, copy the symbol first
 *  before apply.
 *
 * \param sym the symbol to apply
 * \param name the name of symbol
 * \param num_args number of arguments
 * \param keys the key of keyword args (optional)
 * \param symbols arguments to sym
 * \return 0 when success, -1 when failure happens
 */
int NNSymbolCompose(SymbolHandle in,
                             const char* name,
                             nn_uint num_args,
                             const char** in,
                             SymbolHandle* in);

// Graph IR API
/*!
 * \brief create a graph handle from symbol
 * \param symbol The symbol representing the graph.
 * \param graph The graph handle created.
 * \return 0 when success, -1 when failure happens
 */
int NNGraphCreate(SymbolHandle in, GraphHandle *out);
/*!
 * \brief free the graph handle
 * \param handle The handle to be freed.
 */
int NNGraphFree(GraphHandle handle);
/*!
 * \brief Get a new symbol from the graph.
 * \param graph The graph handle.
 * \param symbol The corresponding symbol
 * \return 0 when success, -1 when failure happens
 */
int NNGraphGetSymbol(GraphHandle graph, SymbolHandle *out);

/*!
 * \brief Get Set a attribute in json format.
 * This feature allows pass graph attributes back and forth in reasonable speed.
 *
 * \param handle The graph handle.
 * \param key The key to the attribute.
 * \param json_value The value need to be in format [type_name, value],
 *  Where type_name is a registered type string in C++ side via DMLC_JSON_ENABLE_ANY.
 * \return 0 when success, -1 when failure happens
 */
int NNGraphSetJSONAttr(GraphHandle handle,
                                const char* key,
                                const char* json_value);


/*!
 * \brief Get a serialized attrirbute from graph.
 * This feature allows pass graph attributes back and forth in reasonable speed.
 *
 * \param handle The graph handle.
 * \param key The key to the attribute.
 * \param json_out The result attribute, can be NULL if the attribute do not exist.
 *  The json_out is an array of [type_name, value].
 *  Where the type_name is a registered type string in C++ side via DMLC_JSON_ENABLE_ANY.
 * \param success Whether the result is contained in out.
 * \return 0 when success, -1 when failure happens
 */
int NNGraphGetJSONAttr(SymbolHandle in,
                                const char* key,
                                const char** out,
                                int *out);

/*!
 * \brief Set a attribute whose type is std::vector<NodeEntry> in c++
 * This feature allows pass List of symbolic variables for gradient request.
 *
 * \note This is beta feature only used for test purpos
 *
 * \param handle The graph handle.
 * \param key The key to the attribute.
 * \param list The symbol whose outputs represents the list of NodeEntry to be passed.
 * \return 0 when success, -1 when failure happens
 */
int NNGraphSetNodeEntryListAttr_(GraphHandle handle,
                                          const char* key,
                                          SymbolHandle in);
/*!
 * \brief Apply passes on the src graph.
 * \param src The source graph handle.
 * \param num_pass The number of pass to be applied.
 * \param pass_names The names of the pass.
 * \param dst The result graph.
 * \return 0 when success, -1 when failure happens
 */
int NNGraphApplyPasses(GraphHandle src,
                                nn_uint num_pass,
                                const char** in,
                                GraphHandle *out);
