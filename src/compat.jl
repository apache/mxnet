# this file contains code used for enabling backward compatibility with 0.5

# have to import base dotted operators if in 0.5
if VERSION < v"0.6.0-dev"
    import Base: .+, .-, .*, ./, .^
end


# this is for declaring broadcasted functions in 0.5
# TODO this macro should be removed when 0.5 support is dropped
macro compatdot(fblock)
    if VERSION ≥ v"0.6.0-dev"
        return esc(fblock)
    end
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
    if VERSION ≥ v"0.6.0-dev"
        esc(:(broadcast(*, $expr1, $expr2)))
    else
        esc(:($expr1 .* $expr2))
    end
end
