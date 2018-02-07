# A validation study for the application of quantile regression neural network to Bayesian remote sensing retrievals

This repository contains a study that investigates the applicability of quantile
regression neural networks (QRNN) for retrieving Bayesian a posteriori distributions
of remote sensing retrievals.

## Summary

The aim of the study is to assess the capability of quantile regression neural
networks to estimate the a posteriori distribution of Bayesian data retrievals.
To this end a synthetic retrieval of integrated column water vapor (CWV) from
passive microwave satellite observations has been setup and simulated. The
retrieval is based on a climatology fitted to ERA Interim observations and uses
the ARTS radiative transfer simulator for the generation of training and test
data. The probabilistic predictions obtained from the QRNN are compared to
retrievals performed using MCMC simulations as well as another state-of-the-art
Bayesian retrieval method based on importance sampling of sample from a
retrieval database (BMCI)

## Files

- ~utils.py~: Contains commonly used code for the retrieval setup as well as
   loading and saving of data.
- ~matplotlib_settings.py~: Python module containing code used to setup the Matplotlib
   plotting environment.

