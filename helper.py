import calibr8
import numpy


def infer_independent(
    *,
    model: calibr8.CalibrationModel,
    d: numpy.ndarray,
    y: numpy.ndarray,
    lower: float,
    upper: float,
    steps: int = 300,
    ci_prob: float = 1,
) -> calibr8.ContinuousUnivariateInference:
    """Infer the independent variable from observations at different dilution factors.

    Parameters
    ----------
    model
        A fitted calibration model.
    d
        Vector of dilution factors.
    y
        Vector of observations with the same length as ``d``.
    lower
        Lower limit of the independent variable.
    upper
        Upper limit of the independent variable.
    steps
        Resolution of PDF evaluations to be returned.
    ci_prob
        Probability level for ETI and HDI credible intervals.
        If 1 (default), the complete interval [upper,lower] will be returned,
        else the PDFs will be trimmed to the according probability interval;
        float must be in the interval (0,1].

    Returns
    -------
    pst
        Result of the numeric posterior calculation.
    """
    def likelihood_dilution(x, y, d, model):
        LLs = [model.loglikelihood(x=x / d_i, y=y_i) for d_i, y_i in zip(d, y)]
        return numpy.exp(numpy.sum(LLs, axis=0))

    def likelihood_wrapper(*, x, y, scan_x: bool = False):
        if scan_x:
            # All elements in y are relevant _for each_ element in x
            try:
                # Try to calculate everything in one step without iterating
                return likelihood_dilution(x[..., None], y, d, model)
            except:
                # Broadcasting failed... We'll need to loop it
                return [likelihood_dilution(xi, y, d, model) for xi in x]

        # All elements in x correspond to one element in y
        return likelihood_dilution(x, y, d, model)

    posterior = calibr8.infer_independent(
        likelihood=likelihood_wrapper,
        y=y,
        lower=lower,
        upper=upper,
        steps=steps,
        ci_prob=ci_prob,
    )
    return posterior
