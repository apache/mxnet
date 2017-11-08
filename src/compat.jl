# this file contains code used for enabling backward compatibility with 0.5

# have to import base dotted operators if in 0.5


# this is for declaring broadcasted functions in 0.5
# TODO this macro should be removed when 0.5 support is dropped
macro compatdot(fblock)
    return esc(fblock)
    @capture(fblock, function Base.broadcast(::typeof(op_), args__)
                        body_
                     end)
    opdot = Symbol(string('.',op))
    esc(quote
        function $opdot($(args...))
            $body
        end
    end)
end

macro compatmul(expr1, expr2)
    esc(:(broadcast(*, $expr1, $expr2)))
end
