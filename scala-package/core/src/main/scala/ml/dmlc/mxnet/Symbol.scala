package ml.dmlc.mxnet

class Symbol {
  /**
   * List all the arguments in the symbol.
   * @return Array of all the arguments.
   */
  def listArguments(): Array[String] = ???

  /**
   * List all auxiliary states in the symbool.
   * @return The names of the auxiliary states.
   * Notes
   * -----
   * Auxiliary states are special states of symbols that do not corresponds to an argument,
   * and do not have gradient. But still be useful for the specific operations.
   * A common example of auxiliary state is the moving_mean and moving_variance in BatchNorm.
   * Most operators do not have Auxiliary states.
   */
  def listAuxiliaryStates(): Array[String] = ???
}
