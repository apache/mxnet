module TestVisualize
using MXNet
using Base.Test

using ..Main: mlp2

################################################################################
# Test Implementations
################################################################################

function test_basic()
  info("Visualize::basic")

  mlp = mlp2()
  
  # Order of elements or default color values can change, but length of the output should be more or less stable
  @test length(mx.to_graphviz(mlp)) == length(
"""
digraph "Network Visualization" {
node [fontsize=10];
edge [fontsize=10];
"fc1"  [label="fc1\\nFullyConnected\\nnum-hidden=1000",style="rounded,filled",fixedsize=true,width=1.3,fillcolor="#fb8072",shape=box,penwidth=2,height=0.8034,color="#941305"];
"activation0"  [label="activation0\\nActivation\\nact-type=relu",style="rounded,filled",fixedsize=true,width=1.3,fillcolor="#ffffb3",shape=box,penwidth=2,height=0.8034,color="#999900"];
"fc2"  [label="fc2\\nFullyConnected\\nnum-hidden=10",style="rounded,filled",fixedsize=true,width=1.3,fillcolor="#fb8072",shape=box,penwidth=2,height=0.8034,color="#941305"];
"activation0" -> "fc1"  [arrowtail=open,color="#737373",dir=back];
"fc2" -> "activation0"  [arrowtail=open,color="#737373",dir=back];
}
""")
end
################################################################################
# Run tests
################################################################################
test_basic()
end
