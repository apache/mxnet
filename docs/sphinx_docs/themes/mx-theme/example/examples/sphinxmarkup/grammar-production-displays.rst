===========================
Grammar production displays
===========================

.. code-block:: rst

    .. productionlist::
        try_stmt: try1_stmt | try2_stmt
        try1_stmt: "try" ":" `suite`
                    : ("except" [`expression` ["," `target`]] ":" `suite`)+
                    : ["else" ":" `suite`]
                    : ["finally" ":" `suite`]
        try2_stmt: "try" ":" `suite`
                    : "finally" ":" `suite`

.. productionlist::
   try_stmt: try1_stmt | try2_stmt
   try1_stmt: "try" ":" `suite`
            : ("except" [`expression` ["," `target`]] ":" `suite`)+
            : ["else" ":" `suite`]
            : ["finally" ":" `suite`]
   try2_stmt: "try" ":" `suite`
            : "finally" ":" `suite`
