class Model:
    """
    Base class for machine learning models

    Implementations of models should sublcass this class and implement the
    methods `learn()` and `infer()`.
    """

    def learn(input_data, labels):
        """
        Learn a representation from the given input data and labels.

        :param input_data: Input data to learn from, a pytorch tensor.
        :param labels: Lablels for the input, a pytorch tensor.
        """
        pass

    def infer(input_data):
        """
        Use the model to perform inference on input.

        :param input_data: The input to perform inference on. Pytorch tensor.
        :returns: A pytorch tensor containing inference outputs.
        """
        pass
