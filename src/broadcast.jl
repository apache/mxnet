using TakingBroadcastSeriously: Broadcasted, unwrap

for f in :[%,
           tan, asin, acos, atan,
           sinh, cosh, tanh, asinh, acosh, atanh,
           min, max,
           hypot].args
  # copy from TakingBroadcastSeriously
  @eval Base.$f(a::Broadcasted...) = Broadcasted(broadcast_($f, unwrap.(a)...))
  @eval Base.$f(a::Broadcasted, b) = Broadcasted(broadcast_($f, unwrap(a), b))
  @eval Base.$f(b, a::Broadcasted) = Broadcasted(broadcast_($f, b, unwrap(a)))
end

for f in :[σ, sigmoid, relu, softmax, log_softmax].args
  # copy from TakingBroadcastSeriously
  @eval $f(a::Broadcasted...) = Broadcasted(broadcast_($f, unwrap.(a)...))
  @eval $f(a::Broadcasted, b) = Broadcasted(broadcast_($f, unwrap(a), b))
  @eval $f(b, a::Broadcasted) = Broadcasted(broadcast_($f, b, unwrap(a)))
end
