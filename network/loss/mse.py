

# Mean squared error over the outputs. If only one output node then squared error.
class MSE:
    @staticmethod
    # returns mean squared errors as batch tensor, and also mean of mean squared errors for entire batch
    def error(outputs, targets):
        batch_size = len(outputs)
        number_of_output_nodes = len(outputs[0])
        frac = 1 / number_of_output_nodes
        squared_errors = (outputs - targets)**2
        mses = frac * (squared_errors.sum(axis=1))
        return mses.sum() / batch_size


    @staticmethod
    def derivative(outputs, targets):
        number_of_output_nodes = len(outputs[0])
        frac = 2 / number_of_output_nodes
        return frac * (outputs - targets)
