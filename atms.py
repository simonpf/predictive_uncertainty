""" ARTS Setup for atms sfimulations.

This file contains functions for the setup of the ARTS simulations used tp
produce the training data and perform the MCMC retrievals.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from   netCDF4 import Dataset
from typhon.physics import density
from typhon.arts.workspace import arts_agenda

################################################################################
# Conversion Functions
################################################################################

def vmr2cd(ws):
    """
    Get the total integrated column water value from the ARTS arts workspace.
    """
    p = ws.p_grid.value
    t = ws.t_field.value.ravel()
    z = ws.z_field.value.ravel()
    vmr = ws.vmr_field.value[0,:,:,:].ravel()
    rho = density(p, t)
    mr  = vmr * 18.015 / 28.97
    return np.trapz(mr * rho, z)

def mmr2vmr(ws, x, species = "h2o"):
    """
    Convert mass mixing ratio to volume mixing ratio.
    """
    m_air =  28.971

    if species == "h2o":
        mm = 18.0
    elif species == "o3":
        mm = 48.0
    else:
        Exception("Species not supported.")

    return x / mm * m_air

################################################################################
# ARTS Setup
################################################################################

def surface_fastem(ws):
    ws.specular_losCalc()
    ws.InterpAtmFieldToPosition( out=ws.surface_skin_t, field=ws.t_field )
    ws.surfaceFastem(salinity = ws.salinity,
                     wind_speed = ws.wind_speed,
                     wind_direction = ws.wind_direction,
                     transmittance = ws.transmittance)

def setup_atmosphere(ws):
    """
    This functions performs the setup of the atmospheric state represented by
    the given ARTS workspace object to the mean atmospheric state that was
    obtained from ERA-Interim data.
    """

    # We need to reverse the order of the mean states since the
    # since this is required by ARTS.
    p  = np.load("data/p_grid.npy").ravel()[::-1] * 100.0
    t  = np.load("data/t_mean.npy").ravel()[::-1]
    q  = np.load("data/q_mean.npy").ravel()[::-1]

    ws.execute_controlfile("general/general.arts")
    ws.execute_controlfile("general/continua.arts")
    ws.execute_controlfile("general/agendas.arts")
    ws.execute_controlfile("general/planet_earth.arts")
    ws.Copy(ws.propmat_clearsky_agenda, ws.propmat_clearsky_agenda__LookUpTable)
    ws.Copy(ws.abs_xsec_agenda, ws.abs_xsec_agenda__noCIA)
    ws.Copy(ws.iy_main_agenda, ws.iy_main_agenda__Emission)

    # Salinity, ARTS default
    ws.NumericCreate("salinity")
    ws.salinity = 0.035

    # Wind speed
    ws.NumericCreate("wind_speed")
    ws.wind_speed = 0.0

    # Wind direction
    ws.NumericCreate("wind_direction")
    ws.wind_direction = 0.0

    # Trasmittance for FASTEM
    ws.VectorCreate("transmittance")

    # Agenda for scalar gas absorption calculation
    ws.Copy(ws.abs_xsec_agenda, ws.abs_xsec_agenda__noCIA)

    # Surface
    ws.Copy(ws.iy_surface_agenda, ws.iy_surface_agenda__UseSurfaceRtprop )
    surface_agenda = arts_agenda(surface_fastem)
    ws.Copy(ws.surface_rtprop_agenda, surface_agenda)

    # (standard) emission calculation
    ws.Copy(ws.iy_main_agenda, ws.iy_main_agenda__Emission )

    # cosmic background radiation
    ws.Copy(ws.iy_space_agenda, ws.iy_space_agenda__CosmicBackground )

    # sensor-only path
    ws.Copy(ws.ppath_agenda, ws.ppath_agenda__FollowSensorLosPath )

    # no refraction
    ws.Copy(ws.ppath_step_agenda, ws.ppath_step_agenda__GeometricPath )

    # Set propmat_clearsky_agenda to use lookup table
    ws.Copy(ws.propmat_clearsky_agenda, ws.propmat_clearsky_agenda__LookUpTable )

    ws.stokes_dim = 1
    ws.iy_unit = "PlanckBT"
    ws.cloudboxOff()

    ws.abs_speciesSet(species=["H2O"])
    ws.ReadXML(ws.abs_lines, "instruments/metmm/abs_lines_metmm.xml.gz")
    ws.abs_lines_per_speciesCreateFromLines()

    ws.atmosphere_dim = 1

    # P Grid
    ws.p_grid    = p

    # Temperature
    ws.t_field = np.zeros((p.shape[0], 1, 1))
    ws.t_field.value[:, 0, 0] = t

    # VMR Fields
    ws.vmr_field = np.zeros((1, p.shape[0], 1, 1))

    q = mmr2vmr(ws, q, "h2o")
    ws.vmr_field.value[0, :, 0, 0] = q

    # z field
    ws.VectorCreate("z_vector")
    ws.ZFromPSimple(z_grid = ws.z_vector, p_grid = ws.p_grid)
    z = ws.z_vector.value
    ws.z_field = z.reshape(-1, 1, 1)

    ws.z_surface = ws.z_field.value[0]

    ws.jacobianOff()

def setup_sensor(ws, channels = [-1]):
    """
    This functions performs the setup of the sensor for the simulations of
    brightness temperatures. This uses a slightly modified version of the
    met_mm package provided within ARTS. The changes are contained in a
    modified sensor_atms controlfile which is part of this repository. The
    changes reduces the frequency sampling inside the ATMS channels in order
    to speed up the simulations.
    """
    ws.ArrayOfIndexCreate("channels")
    ws.channels = channels
    ws.ArrayOfIndexCreate("viewing_angles")
    ws.viewing_angles = [47]

    ws.sensor_pos  = np.array([[850e3]]) # 850km
    ws.sensor_time = np.array([0.0])
    ws.sensor_los  = np.array([[180.0]]) # nadir viewing

    ws.IndexCreate("met_mm_accuracy")
    ws.met_mm_accuracy = 1

    ws.execute_controlfile("instruments/metmm/sensor_descriptions/"
                           "prepare_metmm.arts")
    ws.execute_controlfile("sensor_atms.arts")
    ws.execute_controlfile("instruments/metmm/sensor_descriptions/"
                           "apply_metmm.arts")
    ws.transmittance = np.ones(ws.f_grid.value.shape)

def checks(ws):
    """
    Performs ARTS checks required before being able to simulating
    brightness spectra.
    """
    ws.atmfields_checkedCalc( bad_partition_functions_ok = 1 )
    ws.abs_xsec_agenda_checkedCalc()
    ws.abs_lookupSetup()
    ws.abs_lookupCalc()
    ws.surface_scalar_reflectivity = np.array([0.5])
    ws.propmat_clearsky_agenda_checkedCalc()
    ws.atmgeom_checkedCalc()
    ws.cloudbox_checkedCalc()
    ws.sensor_checkedCalc()
    ws.atmgeom_checkedCalc()

################################################################################
# Auxiliary Functions
################################################################################

def softplus(x):
    return np.log(1.0 + np.exp(x))

class StateDistribution:
    def __init__(self):
        self.q_log_mean = np.load("data/q_log_mean.npy").ravel()
        self.q_log_cov  = np.load("data/q_log_cov.npy")
        self.t_mean     = np.load("data/t_mean.npy").ravel()
        self.t_cov      = np.load("data/t_cov.npy")
        self.qt_mean    = np.load("data/qt_mean.npy")
        self.qt_cov     = np.load("data/qt_cov.npy")

    def sample(self, ws = None):
        q = np.exp(np.random.multivariate_normal(self.q_log_mean, self.q_log_cov))
        q = q.ravel()[::-1]
        t = np.random.multivariate_normal(self.t_mean, self.t_cov)
        t = t.ravel()[::-1]
        qt = np.random.multivariate_normal(self.qt_mean, self.qt_cov)

        if ws:
            ws.t_field.value[:, 0, 0] = qt[:14:-1]
            q_vmr = mmr2vmr(ws, np.exp(qt[14::-1]), "h2o")
            ws.vmr_field.value[0, :, 0, 0] = q_vmr

        return (q_vmr, t)

    def sample_factors(self):
        f_q = np.random.multivariate_normal(self.q_log_mean, self.q_log_cov)
        f_t = np.random.multivariate_normal(self.t_mean, self.t_cov)
        f_qt = np.random.multivariate_normal(self.qt_mean, self.qt_cov)
        return f_qt

    def a_priori(self, ws = None):
        q  = np.exp(self.qt_mean[14::-1]).ravel()
        t  = self.qt_mean[:14:-1]

        if ws:
            q_vmr = mmr2vmr(ws, q, "h2o")
            ws.vmr_field.value[0, :, 0, 0] = np.copy(q_vmr)
            ws.t_field.value[:, 0, 0] = np.copy(t)

        return (q_vmr, t)

def create_output_file(filename, n_channels, profile_size):
    """
    Create the netcdf output file for the results of the MCMC simulations.
    """
    if not os.path.isfile(filename):
        root_group = Dataset(filename, mode="w", format="NETCDF4")
        d_step = root_group.createDimension("time", None)
        d_step = root_group.createDimension("step", None)
        d_channel = root_group.createDimension("channel", n_channels)
        d_z = root_group.createDimension("altitude", profile_size)
        v_y_true = root_group.createVariable("y", "f8", ("time", "channel"))
        v_cwv_true = root_group.createVariable("cwv_true", "f8", ("time",))
        v_cwv = root_group.createVariable("cwv", "f8", ("time", "step",))
        v_h2o = root_group.createVariable("h2o", "f8", ("time",
                                                        "step",
                                                        "altitude"))
    else:
        try:
            print("loading existing file: " + filename)
            root_group = Dataset(filename, mode="a", format="NETCDF4")
            d_time = root_group.dimensions["time"]
            d_step = root_group.dimensions["step"]
            d_z    = root_group.dimensions["altitude"]

            v_y_true = root_group.variables["y"]
            v_cwv_true = root_group.variables["cwv_true"]
            v_h2o = root_group.variables["h2o"]
            v_cwv = root_group.variables["cwv"]
        except:
            os.remove(filename)
            raise Exception("File " + str(filename) + " exists but is "
                            " inconsistent with expected file format.")
    return (root_group, v_y_true, v_cwv_true, v_cwv, v_h2o)

def load_file(filename):
    """
    Open a netcdf file containing MCMC simulations and return a tuple containing
    root_group.
    """
    try:
        root_group = Dataset(filename, mode="r+", format="NETCDF4")
        d_time = root_group.dimensions["time"]
        d_step = root_group.dimensions["step"]
        d_z    = root_group.dimensions["altitude"]

        v_y_true = root_group.variables["y"]
        v_cwv_true = root_group.variables["cwv_true"]
        v_h2o = root_group.variables["h2o"]
        v_cwv = root_group.variables["cwv"]
    except:
        raise Exception("File " + str(filename) + " is inconsistent with "
                        " expected file format or does not exit.")
    return (root_group, v_y_true, v_cwv_true, v_cwv, v_h2o)
