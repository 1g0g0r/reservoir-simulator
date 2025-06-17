"""Модуль содержит реализацию решения сеточных уравнений."""
from .Flows_in_cells import flows_in_cells
from .Pressure import calc_pressure
from .Qp_m_k_fi import calc_qp_m_k_fi
from .Saturation import saturation_equation, saturation_well
from .Temperature import temperature_equation, temperature_well
from .Velocity_h import calc_velocitys_h
from .Wps_Wp import wps_wp_equation, wps_wp_wells
