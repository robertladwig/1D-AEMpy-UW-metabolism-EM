#new
import numpy as np
import pandas as pd
import os
#import scipy
from math import pi, exp, sqrt
from scipy.interpolate import interp1d
from copy import deepcopy
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from numba import jit
from functools import reduce

#os.chdir("/home/robert/Projects/1D-AEMpy/src")
#os.chdir("C:/Users/ladwi/Documents/Projects/R/1D-AEMpy/src")
#os.chdir("D:/bensd/Documents/Python_Workspace/1D-AEMpy/src")
#os.chdir("/Users/emmamarchisin/Desktop/Research/Code/1D-AEMpy-UW-metabolism-EM/src")
os.chdir('/Users/au740615/Documents/projects/1D-AEMpy-UW-metabolism-EM/src')
from processBased_lakeModel_functions import get_hypsography, provide_meteorology, initial_profile, run_wq_model, wq_initial_profile, provide_phosphorus, provide_carbon, do_sat_calc, calc_dens,atmospheric_module, get_secview, get_lake_config, get_model_params, get_run_config, get_ice_and_snow , get_num_data_columns#, heating_module, diffusion_module, mixing_module, convection_module, ice_module



Start = datetime.datetime.now()
num_lakes = get_num_data_columns(
    "../input/ME/lake_config.csv", "Zmax"
)

for lake_num in range(1, num_lakes + 1):

   
    lake_config = get_lake_config( # RL: added Longitdue, Latitude and Elevation
        "../input/ME/lake_config.csv", lake_num
    )
    model_params = get_model_params(
        "../input/ME/model_params.csv", lake_num
    )
    run_config = get_run_config(
        "../input/ME/run_config.csv", lake_num
    )
    ice_and_snow = get_ice_and_snow(
        "../input/ME/ice_and_snow.csv", lake_num
    )
    windfactor = float(lake_config["WindSpeed"])
    zmax = lake_config['Zmax']
    nx = int(run_config["nx"])# number of layers we will have
    dt = float(run_config["dt"])# 24 hours times 60 min/hour times 60 seconds/min to convert s to day
    dx = float(run_config["dx"]) # spatial step
    ## area and depth values of our lake 
    area, depth, volume, hypso_weight = get_hypsography(hypsofile = '../input/ME/bathymetry.csv',#'../input/Peter Lake/bathymetry.csv',
                            dx = dx, nx = nx, outflow_depth=float(lake_config["outflow_depth"]))
    #area, depth, volume = get_hypsography(hypsofile = '../input/bathymetry.csv',
      #                      dx = dx, nx = nx)

                
    

    ## time step discretization 

    #get start time from input file
    desired_start = pd.Timestamp(run_config["start_time"])  
    desired_end = pd.Timestamp(run_config["end_time"])  
    
    #find the matching index in the meteo file
    # startTime = meteo_all.index[meteo_all['date'] == desired_start][0]
    # startTime = startTime
    # #get the date from that index
    # startingDate = meteo_all.loc[startTime, 'date']
    
    startTime = 1 # RL: SOMEONE SHOULD THINK ABOUT THIS MORE DEEPLY!
    startingDate = desired_start
    # n_years = run_config['n_years'] # RL: this was not in config file
    
    n_days = (desired_end - desired_start).days + (desired_end - desired_start).seconds/86400 #(desired_end - desired_start).days
    
    hydrodynamic_timestep = 24 * dt
    total_runtime =  (n_days) * hydrodynamic_timestep/dt  
    
    endTime =  (startTime + total_runtime) 
  
    endingDate = desired_end #  meteo_all['date'][(endTime-1)]

    print ("starting date", startingDate)
    print ("starting desored", desired_start)
    print ("starting time", startTime)

    print ("ending time", endTime)
    times = pd.date_range(startingDate, endingDate, freq='H')

    nTotalSteps = int(total_runtime)

    meteo_all = provide_meteorology(meteofile = run_config["meteo_ini_file"], 
                    windfactor = windfactor, lat = lake_config["Latitude"], lon = lake_config["Longitude"], elev = lake_config["Elevation"],
                    startDate = startingDate)

    pd.DataFrame(meteo_all).to_csv("../input/ME/NLDAS-ME-meteo16-24.csv", index = False)
                     
    atm_flux_output = np.zeros(nTotalSteps,) 
    u_ini = initial_profile(initfile = run_config["u_ini_file"], nx = nx, dx = dx,
                     depth = depth,
                     startDate = startingDate) 
    wq_ini = wq_initial_profile(initfile = run_config["wq_ini_file"], nx = nx, dx = dx,
                     depth = depth, 
                     volume = volume,
                     startDate = startingDate)
    tp_boundary = provide_phosphorus(tpfile =  run_config["tp_ini_file"], 
                                 startingDate = startingDate,
                                 startTime = startTime)
    carbon = provide_carbon(ocloadfile =  run_config["oc_load_file"], # RL: carbon driver?
                                 startingDate=startingDate,
                                 startTime = startTime)
    carbon = carbon.dropna(subset=['oc'])


    res = run_wq_model(
        # RUNTIME CONFIG
        lake_num=lake_num,
        startTime=startingDate,
        endTime=endingDate,
        nx=run_config["nx"],
        dt=run_config["dt"],
        dx=run_config["dx"],
        timelabels=times,  # = run_config["times"]
        pgdl_mode=run_config["pgdl_mode"],
        training_data_path=run_config["training_data_path"],
        diffusion_method=run_config["diffusion_method"],
        scheme=run_config["scheme"],

        # LAKE CONFIG
        area=area,  # already read
        volume=volume,  # already read
        depth=depth,  # already read
        zmax=lake_config['Zmax'],
        outflow_depth=lake_config['outflow_depth'],
        mean_depth=sum(volume) / max(area),
        hypso_weight=hypso_weight,
        altitude=lake_config['Elevation'],
        #elev=altitude,
        lat=lake_config['Latitude'],
        long=lake_config['Longitude'],

        # MODEL PARAMS - initial conditions
        u=deepcopy(u_ini),  # already read
        o2=deepcopy(wq_ini[0]),  # already read
        docr=deepcopy(wq_ini[1])*.75, #* 1.3,
        docl=deepcopy(wq_ini[1])*.25,#1.0 * volume,
        pocr=0.5 * volume,
        pocl=0.5 * volume,

        # meteorology & boundary forcing
        daily_meteo=meteo_all,
        secview=None,
        phosphorus_data=tp_boundary,
        oc_load_input=carbon,

        # ice & snow dynamics
        ice=ice_and_snow["ice"],
        Hi=ice_and_snow["Hi"],
        Hs=ice_and_snow["Hs"],
        Hsi=ice_and_snow["Hsi"],
        iceT=ice_and_snow["iceT"],
        supercooled=ice_and_snow["supercooled"],
        dt_iceon_avg=ice_and_snow["dt_iceon_avg"],
        Ice_min=ice_and_snow["Ice_min"],
        KEice=ice_and_snow["KEice"],
        rho_snow=ice_and_snow["rho_snow"],
    

        # mixing and physical transport
        km=model_params["km"],
        k0=model_params["k0"],
        weight_kz=model_params["weight_kz"],
        piston_velocity=model_params["piston_velocity"]/86400,
        Cd=model_params["Cd"],
        hydro_res_time_hr=model_params["hydro_res_time"]*8760,
        W_str=(
            None if pd.isna(model_params["W_str"])
            else model_params["W_str"]
        ),
        denThresh=model_params["denThresh"],

        # light & heat fluxes
        kd_light=model_params["kd_light"],
        light_water=model_params["light_water"],
        light_doc=model_params["light_doc"],
        light_poc=model_params["light_poc"],
        albedo=lake_config["Albedo"],
        eps=model_params["eps"],
        emissivity=model_params["emissivity"],
        sigma=model_params["sigma"],
        sw_factor=model_params["sw_factor"],
        wind_factor=model_params["wind_factor"],
        at_factor=model_params["at_factor"],
        turb_factor=model_params["turb_factor"],
        Hgeo=model_params["Hgeo"],

        # biogeochemical params
        resp_docr=model_params["resp_docr"]/86400,
        resp_docl=model_params["resp_docl"]/86400,
        resp_pocr=model_params["resp_poc"]/86400,
        resp_pocl=model_params["resp_poc"]/86400,
        resp_poc=model_params["resp_poc"]/86400,
        sed_sink=model_params["sed_sink"]/86400,
        settling_rate=model_params["settling_rate"]/86400,
        sediment_rate=model_params["sediment_rate"]/86400,
        theta_npp=model_params["theta_npp"],
        theta_r=model_params["theta_r"],
        conversion_constant=model_params["conversion_constant"],
        k_half=model_params["k_half"],
        p_max=model_params["p_max"]/86400,
        IP=model_params["IP"]/86400,
        f_sod=model_params["f_sod"],
        d_thick=model_params["d_thick"],
        LIGHTUSEBYPHOTOS = model_params["LIGHTUSEBYPHOTOS"],

        # carbon pool partitioning
        prop_oc_docr=model_params["prop_oc_docr"],
        prop_oc_docl=model_params["prop_oc_docl"],
        prop_oc_pocr=model_params["prop_oc_pocr"],
        prop_oc_pocl=model_params["prop_oc_pocl"],

        # general physical constants
        p2=model_params["p2"],
        B=model_params["B"],
        g=model_params["g"],
        meltP=model_params["meltP"],
    )

   # atm_flux=atm_flux)

temp=  res['temp']
o2=  res['o2']
docr=  res['docr']
docl =  res['docl']
pocr=  res['pocr']
pocl=  res['pocl']
diff =  res['diff']
avgtemp = res['average'].values
temp_initial =  res['temp_initial']
temp_heat=  res['temp_heat']
temp_diff=  res['temp_diff']
temp_mix =  res['temp_mix']
temp_conv =  res['temp_conv']
temp_ice=  res['temp_ice']
meteo=  res['meteo_input']
buoyancy = res['buoyancy']
icethickness= res['icethickness']
snowthickness= res['snowthickness']
snowicethickness= res['snowicethickness']
npp = res['npp']
docr_respiration = res['docr_respiration']
docl_respiration = res['docl_respiration']
poc_respiration = res['poc_respiration']
kd = res['kd_light']
secchi = res['secchi']
thermo_dep = res['thermo_dep']
energy_ratio = res['energy_ratio']
atm_flux_output=res['atm_flux_output']

SW_flux = meteo[4, ]

End = datetime.datetime.now()
print(End - Start)

####diagnistic graphs###
light=meteo_all['Shortwave_Radiation_Downwelling_wattPerMeterSquared'].iloc[int(startTime):int(endTime)+1].reset_index(drop=True)
wind=meteo_all['Ten_Meter_Elevation_Wind_Speed_meterPerSecond'].iloc[int(startTime):int(endTime)+1].reset_index(drop=True)
precip=meteo_all['Precipitation_millimeterPerDay'].iloc[int(startTime):int(endTime)+1].reset_index(drop=True)
air_temp=meteo_all['Air_Temperature_celsius'].iloc[int(startTime):int(endTime)+1].reset_index(drop=True)
#times2 = meteo_all[0]['datetime']


fig,axis=plt.subplots(4,1,figsize=(12,6),sharex=True)
axis[0].plot(times, light, color='goldenrod')
axis[0].set_ylabel('Light (µmol/m²/s)')
axis[0].set_title('Meteorology: Light and Wind')

axis[1].plot(times, wind, color='steelblue')
axis[1].set_ylabel('Wind (m/s)')
axis[1].set_xlabel('Time')

axis[2].plot(times, precip, color='red')
axis[2].set_ylabel('Precip(mm/d)')
axis[2].set_xlabel('Time')

axis[3].plot(times, air_temp, color='red')
axis[3].set_ylabel('Air Temp (degC)')
axis[3].set_xlabel('Time')

plt.tight_layout()
plt.show()

# plt.plot(icethickness[0])
# plt.show()


# plt.plot(npp[0,:])
# plt.show()

depth1=2 #index for 1m depth
def compute_delta_hourly(var):
    delta=np.empty_like(var)
    delta[0]=np.nan
    delta[1:]=var[1:]-var[:-1]
    return delta

temp_1m=temp[depth1,:]
do_1m=o2[depth1,:]/volume[depth1]
#gpp_1m=(npp[2,:] -1/86400 *(docl[2,:] * docl_respiration[2,:]+ docr[2,:] * docr_respiration[2,:] + pocl[2,:] * poc_respiration[2,:] + pocr[2,:] * poc_respiration[2,:]))*(1/volume[depth1])*3600#/volume[depth1]
#r_1m=1/86400*(docl[2,:] * docl_respiration[2,:]+ docr[2,:] * docr_respiration[2,:] + pocl[2,:] * poc_respiration[2,:] + pocr[2,:] * poc_respiration[2,:])/volume[2]*3600#3600 to get from g/m3/s to g/m3/h
r_1m_hour=((docl[depth1, :] * docl_respiration[depth1, :]) +
    (docr[depth1, :] * docr_respiration[depth1, :]) +
    (pocl[depth1, :] * poc_respiration[depth1, :]) +
    (pocr[depth1, :] * poc_respiration[depth1, :])) / volume[depth1]/24 #g/m3/h
npp_1m_hour=npp[depth1,:]/volume[depth1]/24 #g/m3/h
gpp_1m_hour=npp_1m_hour+r_1m_hour #g/m3/h
atm_1m=atm_flux_output[0,:]/volume[0]/24 #g o2/m3/h
delta_gpp1m=compute_delta_hourly(gpp_1m_hour)
delta_r1m=compute_delta_hourly(r_1m_hour)
delta_atm1m=compute_delta_hourly(atm_1m)  
 
fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
ax[0].plot(times, temp_1m, color='orangered')
ax[0].set_ylabel('Water Temp (°C)')
ax[0].set_title('Water Temperature and DO at 1 m')
ax[1].plot(times, do_1m, color='blue')
ax[1].set_ylabel('Dissolved Oxygen (mg/L)')
ax[1].set_xlabel('Time')
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
ax[0].plot(times, gpp_1m_hour*24, label='GPP', color='green')
#ax[0].set_ylabel('GPP (g/m3/d)')
#ax[0].plot(times, npp[2,:]*24*3600/volume[2], label='NPP', color='black')
ax[0].set_ylabel('GPP (g/m3/d)')
ax[0].legend()
ax[1].plot(times, r_1m_hour*24, label='Respiration (R)', color='red')
ax[1].set_ylabel('R (g/m3/d)')
ax[1].legend()
ax[2].plot(times, atm_1m*24, label='Atmospheric Exchange', color='purple')
ax[2].set_ylabel('Atmospheric Exchange (g/m3/d)')
ax[2].legend()
ax[3].plot(times, delta_gpp1m, label='Δ GPP', color='darkgreen')
ax[3].plot(times, delta_atm1m, label='Δ Atmospheric Exchange', color='indigo')
ax[3].set_ylabel('Hourly Flux Change')
ax[3].set_xlabel('Time')
ax[3].legend()
ax[4].plot(times, delta_r1m, label='Δ R', color='darkred')
ax[4].set_ylabel('Hourly Flux Change')
ax[4].set_xlabel('Time')
ax[4].legend()
plt.tight_layout()
plt.show()


plt.plot(times, energy_ratio[0,:])
plt.ylabel("Energy Ratio", fontsize=15)
plt.xlabel("Time", fontsize=15)   

# heatmap of temps  
N_pts = 6
n_years = float(n_days / 365)


fig, ax = plt.subplots(figsize=(15,5))
sns.heatmap(temp, cmap=plt.cm.get_cmap('Spectral_r'),  xticklabels=1000, yticklabels=2, vmin = 0, vmax = 30)
ax.contour(np.arange(.5, temp.shape[1]), np.arange(.5, temp.shape[0]), calc_dens(temp), levels=[999],
           colors=['black', 'gray'],
           linestyles = 'dotted')
ax.set_ylabel("Depth (m)", fontsize=15)
ax.set_xlabel("Time", fontsize=15)    
ax.collections[0].colorbar.set_label("Water Temperature  (dC)")
xticks_ix = np.array(ax.get_xticks()).astype(int)
time_label = times[xticks_ix]
nelement = len(times)//N_pts
#time_label = time_label[::nelement]
#ax.xaxis.set_major_locator(plt.MaxNLocator(N_pts * n_years))
ax.set_xticklabels(time_label.strftime("%d-%m-%y"), rotation=45, ha = 'right')
yticks_ix = np.array(ax.get_yticks()).astype(int)
depth_label = yticks_ix / 2
ax.set_yticklabels(depth_label, rotation=0)
plt.show()

# breakpoint()
# fig, ax = plt.subplots(figsize=(15,5))
# sns.heatmap(SW_flux, cmap=plt.cm.get_cmap('Spectral_r'),  xticklabels=1000, yticklabels=2, vmin = 0, vmax = 30)
# ax.contour(np.arange(.5, temp.shape[1]), np.arange(.5, temp.shape[0]), calc_dens(temp), levels=[999],
#            colors=['black', 'gray'],
#            linestyles = 'dotted')
# ax.set_ylabel("Depth (m)", fontsize=15)
# ax.set_xlabel("Time", fontsize=15)    
# ax.collections[0].colorbar.set_label("Short-wave flux  (W/m2)")
# xticks_ix = np.array(ax.get_xticks()).astype(int)
# time_label = times[xticks_ix]
# nelement = len(times)//N_pts
# #time_label = time_label[::nelement]
# #ax.xaxis.set_major_locator(plt.MaxNLocator(N_pts * n_years))
# ax.set_xticklabels(time_label.strftime("%d-%m-%y"), rotation=45, ha = 'right')
# yticks_ix = np.array(ax.get_yticks()).astype(int)
# depth_label = yticks_ix / 2
# ax.set_yticklabels(depth_label, rotation=0)
# plt.show()



fig, ax = plt.subplots(figsize=(15,5))
sns.heatmap(diff, cmap=plt.cm.get_cmap('Spectral_r'),  xticklabels=1000, yticklabels=2, vmin = 1e-4, vmax = 1e-2)
ax.contour(np.arange(.5, temp.shape[1]), np.arange(.5, temp.shape[0]), calc_dens(temp), levels=[999],
           colors=['black', 'gray'],
           linestyles = 'dotted')
ax.set_ylabel("Depth (m)", fontsize=15)
ax.set_xlabel("Time", fontsize=15)    
ax.collections[0].colorbar.set_label("Diffusivity  (m2/s)")
xticks_ix = np.array(ax.get_xticks()).astype(int)
time_label = times[xticks_ix]
nelement = len(times)//N_pts
#time_label = time_label[::nelement]
#ax.xaxis.set_major_locator(plt.MaxNLocator(N_pts * n_years))
ax.set_xticklabels(time_label.strftime("%d-%m-%y"), rotation=45, ha = 'right')
yticks_ix = np.array(ax.get_yticks()).astype(int)
depth_label = yticks_ix / 2
ax.set_yticklabels(depth_label, rotation=0)
plt.show()




fig, ax = plt.subplots(figsize=(15,5))
sns.heatmap(np.transpose(np.transpose(o2)/volume), cmap=plt.cm.get_cmap('Spectral_r'),  xticklabels=1000, yticklabels=2, vmin = 0, vmax = 20)
ax.contour(np.arange(.5, temp.shape[1]), np.arange(.5, temp.shape[0]), calc_dens(temp), levels=[999],
           colors=['black', 'gray'],
           linestyles = 'dotted')
ax.set_ylabel("Depth (m)", fontsize=15)
ax.set_xlabel("Time", fontsize=15)    
ax.collections[0].colorbar.set_label("Dissolved Oxygen  (g/m3)")
xticks_ix = np.array(ax.get_xticks()).astype(int)
time_label = times[xticks_ix]
nelement = len(times)//N_pts
#time_label = time_label[::nelement]
#ax.xaxis.set_major_locator(plt.MaxNLocator(N_pts * n_years))
ax.set_xticklabels(time_label.strftime("%d-%m-%y"), rotation=45, ha = 'right')
yticks_ix = np.array(ax.get_yticks()).astype(int)
depth_label = yticks_ix / 2
ax.set_yticklabels(depth_label, rotation=0)
plt.show()


fig, ax = plt.subplots(figsize=(15,5))
sns.heatmap(np.transpose(np.transpose(docl)/volume), cmap=plt.cm.get_cmap('Spectral_r'),  xticklabels=1000, yticklabels=2, vmin = 0, vmax = 7)
ax.contour(np.arange(.5, temp.shape[1]), np.arange(.5, temp.shape[0]), calc_dens(temp), levels=[999],
           colors=['black', 'gray'],
           linestyles = 'dotted')
ax.set_ylabel("Depth (m)", fontsize=15)
ax.set_xlabel("Time", fontsize=15)    
ax.collections[0].colorbar.set_label("DOC-labile  (g/m3)")
xticks_ix = np.array(ax.get_xticks()).astype(int)
time_label = times[xticks_ix]
nelement = len(times)//N_pts
#time_label = time_label[::nelement]
#ax.xaxis.set_major_locator(plt.MaxNLocator(N_pts * n_years)) need back for labels
ax.set_xticklabels(time_label.strftime("%d-%m-%y"), rotation=45, ha = 'right')
yticks_ix = np.array(ax.get_yticks()).astype(int)
depth_label = yticks_ix / 2
ax.set_yticklabels(depth_label, rotation=0)
plt.show()


fig, ax = plt.subplots(figsize=(15,5))
sns.heatmap(np.transpose(np.transpose(docr)/volume), cmap=plt.cm.get_cmap('Spectral_r'),  xticklabels=1000, yticklabels=2, vmin = 0, vmax = 7)
ax.contour(np.arange(.5, temp.shape[1]), np.arange(.5, temp.shape[0]), calc_dens(temp), levels=[999],
           colors=['black', 'gray'],
           linestyles = 'dotted')
ax.set_ylabel("Depth (m)", fontsize=15)
ax.set_xlabel("Time", fontsize=15)    
ax.collections[0].colorbar.set_label("DOC-refractory  (g/m3)")
xticks_ix = np.array(ax.get_xticks()).astype(int)
time_label = times[xticks_ix]
nelement = len(times)//N_pts
#time_label = time_label[::nelement]
#ax.xaxis.set_major_locator(plt.MaxNLocator(N_pts * n_years))
ax.set_xticklabels(time_label.strftime("%d-%m-%y"), rotation=45, ha = 'right')
yticks_ix = np.array(ax.get_yticks()).astype(int)
depth_label = yticks_ix / 2
ax.set_yticklabels(depth_label, rotation=0)
plt.show()


fig, ax = plt.subplots(figsize=(15,5))
sns.heatmap(np.transpose(np.transpose(pocr)/volume), cmap=plt.cm.get_cmap('Spectral_r'),  xticklabels=1000, yticklabels=2, vmin = 0, vmax = 5) #
ax.contour(np.arange(.5, temp.shape[1]), np.arange(.5, temp.shape[0]), calc_dens(temp), levels=[999],
           colors=['black', 'gray'],
           linestyles = 'dotted')
ax.set_ylabel("Depth (m)", fontsize=15)
ax.set_xlabel("Time", fontsize=15)    
ax.collections[0].colorbar.set_label("POC-refractory  (g/m3)")
xticks_ix = np.array(ax.get_xticks()).astype(int)
time_label = times[xticks_ix]
nelement = len(times)//N_pts
#time_label = time_label[::nelement]
#ax.xaxis.set_major_locator(plt.MaxNLocator(N_pts * n_years))
ax.set_xticklabels(time_label.strftime("%d-%m-%y"), rotation=45, ha = 'right')
yticks_ix = np.array(ax.get_yticks()).astype(int)
depth_label = yticks_ix / 2
ax.set_yticklabels(depth_label, rotation=0)
plt.show()

fig, ax = plt.subplots(figsize=(15,5))
sns.heatmap(np.transpose(np.transpose(pocl)/volume), cmap=plt.cm.get_cmap('Spectral_r'),  xticklabels=1000, yticklabels=2, vmin = 0, vmax = 5) #
ax.contour(np.arange(.5, temp.shape[1]), np.arange(.5, temp.shape[0]), calc_dens(temp), levels=[999],
           colors=['black', 'gray'],
           linestyles = 'dotted')
ax.set_ylabel("Depth (m)", fontsize=15)
ax.set_xlabel("Time", fontsize=15)    
ax.collections[0].colorbar.set_label("POC-labile  (g/m3)")
xticks_ix = np.array(ax.get_xticks()).astype(int)
time_label = times[xticks_ix]
nelement = len(times)//N_pts
#time_label = time_label[::nelement]
#ax.xaxis.set_major_locator(plt.MaxNLocator(N_pts * n_years))
ax.set_xticklabels(time_label.strftime("%d-%m-%y"), rotation=45, ha = 'right')
yticks_ix = np.array(ax.get_yticks()).astype(int)
depth_label = yticks_ix / 2
ax.set_yticklabels(depth_label, rotation=0)
plt.show()


fig, ax = plt.subplots(figsize=(15,5))
sns.heatmap((np.transpose(np.transpose(npp)/volume)), cmap=plt.cm.get_cmap('Spectral_r'),  xticklabels=1000, yticklabels=2)
ax.contour(np.arange(.5, temp.shape[1]), np.arange(.5, temp.shape[0]), calc_dens(temp), levels=[999],
           colors=['black', 'gray'],
           linestyles = 'dotted')
ax.set_ylabel("Depth (m)", fontsize=15)
ax.set_xlabel("Time", fontsize=15)    
ax.collections[0].colorbar.set_label("NPP  (g/m3/d)")
xticks_ix = np.array(ax.get_xticks()).astype(int)
time_label = times[xticks_ix]
nelement = len(times)//N_pts
#time_label = time_label[::nelement]
#ax.xaxis.set_major_locator(plt.MaxNLocator(N_pts * n_years))
ax.set_xticklabels(time_label.strftime("%d-%m-%y"), rotation=45, ha = 'right')
yticks_ix = np.array(ax.get_yticks()).astype(int)
depth_label = yticks_ix / 2
ax.set_yticklabels(depth_label, rotation=0)
plt.show()


fig, ax = plt.subplots(figsize=(15,5))
sns.heatmap(docr_respiration , cmap=plt.cm.get_cmap('Spectral_r'),  xticklabels=1000, yticklabels=2, vmin = 0, vmax = 2e-3)
ax.contour(np.arange(.5, temp.shape[1]), np.arange(.5, temp.shape[0]), calc_dens(temp), levels=[999],
           colors=['black', 'gray'],
           linestyles = 'dotted')
ax.set_ylabel("Depth (m)", fontsize=15)
ax.set_xlabel("Time", fontsize=15)    
ax.collections[0].colorbar.set_label("DOCr respiration  (/d)")
xticks_ix = np.array(ax.get_xticks()).astype(int)
time_label = times[xticks_ix]
nelement = len(times)//N_pts
#time_label = time_label[::nelement]
#ax.xaxis.set_major_locator(plt.MaxNLocator(N_pts * n_years))
ax.set_xticklabels(time_label.strftime("%d-%m-%y"), rotation=45, ha = 'right')
yticks_ix = np.array(ax.get_yticks()).astype(int)
depth_label = yticks_ix / 2
ax.set_yticklabels(depth_label, rotation=0)
plt.show()

fig, ax = plt.subplots(figsize=(15,5))
sns.heatmap(docl_respiration , cmap=plt.cm.get_cmap('Spectral_r'),  xticklabels=1000, yticklabels=2, vmin = 0, vmax = 8e-2)
ax.contour(np.arange(.5, temp.shape[1]), np.arange(.5, temp.shape[0]), calc_dens(temp), levels=[999],
           colors=['black', 'gray'],
           linestyles = 'dotted')
ax.set_ylabel("Depth (m)", fontsize=15)
ax.set_xlabel("Time", fontsize=15)    
ax.collections[0].colorbar.set_label("DOCl respiration  (/d)")
xticks_ix = np.array(ax.get_xticks()).astype(int)
time_label = times[xticks_ix]
nelement = len(times)//N_pts
#time_label = time_label[::nelement]
#ax.xaxis.set_major_locator(plt.MaxNLocator(N_pts * n_years))
ax.set_xticklabels(time_label.strftime("%d-%m-%y"), rotation=45, ha = 'right')
yticks_ix = np.array(ax.get_yticks()).astype(int)
depth_label = yticks_ix / 2
ax.set_yticklabels(depth_label, rotation=0)
plt.show()


fig, ax = plt.subplots(figsize=(15,5))
sns.heatmap(poc_respiration , cmap=plt.cm.get_cmap('Spectral_r'),  xticklabels=1000, yticklabels=2, vmin = 0, vmax = 3e-1)
ax.contour(np.arange(.5, temp.shape[1]), np.arange(.5, temp.shape[0]), calc_dens(temp), levels=[999],
           colors=['black', 'gray'],
           linestyles = 'dotted')
ax.set_ylabel("Depth (m)", fontsize=15)
ax.set_xlabel("Time", fontsize=15)    
ax.collections[0].colorbar.set_label("POC respiration  (/d)")
xticks_ix = np.array(ax.get_xticks()).astype(int)
time_label = times[xticks_ix]
nelement = len(times)//N_pts
#time_label = time_label[::nelement]
#ax.xaxis.set_major_locator(plt.MaxNLocator(N_pts * n_years))
ax.set_xticklabels(time_label.strftime("%d-%m-%y"), rotation=45, ha = 'right')
yticks_ix = np.array(ax.get_yticks()).astype(int)
depth_label = yticks_ix / 2
ax.set_yticklabels(depth_label, rotation=0)
plt.show()

# plt.plot(npp[1,1:400]/volume[1] * 86400)
# plt.plot(o2[1,:]/volume[1])
# plt.plot(o2[1,1:(24*14)]/volume[1])
# plt.plot(o2[1,:]/volume[1])
# plt.plot(docl[1,:]/volume[1])
# plt.plot(docr[1,1:(24*10)]/volume[1])
# plt.plot(pocl[0,:]/volume[0])
# plt.plot(pocr[0,:]/volume[0])
# plt.plot(npp[0,:]/volume[0]*86400)
# plt.plot(docl_respiration[0,:]/volume[0]*86400)
# plt.plot(o2[(nx-1),:]/volume[(nx-1)])

plt.plot(o2[1,1:(24*28)]/volume[1]/4, color = 'blue', label = 'O2')
gpp = npp[1,:]/86400 -1/86400 *(docl[1,:] * docl_respiration[1,:]+ docr[1,:] * docr_respiration[1,:] + pocl[1,:] * poc_respiration[1,:] + pocr[1,:] * poc_respiration[1,:])
plt.plot(npp[1,1:(24*28)]/volume[1], color = 'yellow', label = 'GPP') 
plt.plot((-1) * 1/86400*(docl[1,1:(24*28)] * docl_respiration[1,1:(24*28)]+ docr[1,1:(24*28)] * docr_respiration[1,1:(24*28)] + pocl[1,1:(24*28)] * poc_respiration[1,1:(24*28)] + pocr[1,1:(24*28)] * poc_respiration[1,1:(24*28)])/volume[1] * 86400, color = 'red', label = 'R') 
plt.plot(gpp[1:(24*28)]/volume[1] * 86400, color = 'green', label = 'NEP')
plt.legend(loc='best')
plt.show() 



plt.plot(times, kd[0,:])
plt.ylabel("kd (/m)")
plt.show()

plt.plot(times, secchi[0,:])
plt.ylabel("Secchi Depth (m)")
plt.xlabel("Time")
plt.show()

do_sat = o2[0,:] * 0.0
for r in range(0, len(temp[0,:])):
    do_sat[r] = do_sat_calc(temp[2,r], 982.2, 258) 

plt.plot(times, o2[0,:]/volume[0], color = 'blue')
plt.plot(times, do_sat, color = 'red')
plt.show()

plt.plot(times, thermo_dep[0,:]*dx,color= 'blue')
plt.plot(times, temp[0,:] - temp[(nx-1),:], color = 'red')
plt.show()

#Diagnostic graphs at depth 
depths = [1,44]   # Python indices for depth=1 and depth=22
labels = ['Depth 1', 'Depth 22']

doc_total = np.add(docl, docr)
poc_total = np.add(pocl, pocr)
# Plot DO
plt.figure(figsize=(10, 5))
for i, d in enumerate(depths):
    plt.plot(times, o2[d, :] / volume[d], label=f'DO at {labels[i]}', linestyle='-', color=('blue' if d == 1 else 'cyan'))
plt.ylabel("DO (mg/L)")
plt.xlabel("Time")
plt.legend()
plt.title("Dissolved Oxygen (DO)")
plt.show()


plt.figure(figsize=(10, 5))
for i, d in enumerate(depths):
    max_doc = (doc_total[d, :] / volume[d]).max()   # take max over time
    print(f"Depth index {d}: max DOC total = {max_doc:.2f} mg/L")
    plt.plot(times, doc_total[d, :]/volume [d], label=f'DOC at {labels[i]}', linestyle='-', color=('green' if d == 1 else 'lightgreen'))
plt.ylabel("DOC (mg/L)")
plt.xlabel("Time")
plt.ylim(0, 6)
plt.legend()
plt.title("Dissolved Organic Carbon Total (DOC-tot)")
plt.show()

plt.figure(figsize=(10, 5))
for i, d in enumerate(depths):
    max_doc = (docl[d, :] / volume[d]).max()   # take max over time
    print(f"Depth index {d}: max DOCl = {max_doc:.2f} mg/L")
    plt.plot(times, docl[d, :]/volume [d], label=f'DOC at {labels[i]}', linestyle='-', color=('green' if d == 1 else 'lightgreen'))
plt.ylabel("DOC (mg/L)")
plt.xlabel("Time")
plt.ylim(0, 6)
plt.legend()
plt.title("Dissolved Organic Carbon Laible (DOCl)")
plt.show()

plt.figure(figsize=(10, 5))
for i, d in enumerate(depths):
    max_doc = (docr[d, :] / volume[d]).max()   # take max over time
    print(f"Depth index {d}: max DOCr = {max_doc:.2f} mg/L")
    plt.plot(times, docr[d, :]/volume [d], label=f'DOC at {labels[i]}', linestyle='-', color=('green' if d == 1 else 'lightgreen'))
plt.ylabel("DOC (mg/L)")
plt.xlabel("Time")
plt.ylim(0, 6)
plt.legend()
plt.title("Dissolved Organic Carbon Recalcitrant (DOCr)")
plt.show()

# Plot POC
plt.figure(figsize=(10, 5))
for i, d in enumerate(depths):
    plt.plot(times, poc_total[d, :]/volume[d], label=f'POC at {labels[i]}', linestyle='-', color=('orange' if d == 1 else 'gold'))
plt.ylabel("POC (mg/L)")
plt.xlabel("Time")
#plt.ylim(0, 8)
plt.legend()
plt.title("Particulate Organic Carbon (POC)")
plt.show()

#Check against observed data
df_obs=pd.read_csv('../input/mendota_driver_data_v3.csv',  parse_dates=['datetime'])
df_obs['datetime'] = pd.to_datetime(df_obs['datetime'], errors='coerce')
df_obs = df_obs[(df_obs['datetime'] >= startingDate) & (df_obs['datetime'] <= endingDate)]
df_obs_surf_do = df_obs[(df_obs['variable'] == 'do') & (df_obs['depth'] == 1)]
df_obs_surf_do['datetime'] = pd.to_datetime(df_obs_surf_do['datetime'], format = 'mixed')
df_obs_bot_do= df_obs[(df_obs['variable'] == 'do') & (df_obs['depth'] == 22)]

plt.figure(figsize=(10, 5))
plt.plot(times, o2[2,:]/volume[2], color= 'blue', label='1m Modeled DO', linestyle= 'solid')
plt.plot(times, o2[44,:]/volume[44], color= 'blue', label='22m Modeled DO', linestyle= 'dashed')
plt.plot(df_obs_surf_do["datetime"], df_obs_surf_do["observation"], color= 'red', label='1m Observed DO', linestyle= 'solid',marker='o', zorder=5)
plt.plot(df_obs_bot_do["datetime"], df_obs_bot_do["observation"], color= 'red', label='22m Observed DO', linestyle= 'dashed', marker='o', zorder=5)
plt.ylabel("DO (mg/L)", fontsize=15)
plt.xlabel("Time", fontsize=15) 
plt.legend(loc='best')
plt.show()

df_obs_surf_doc = df_obs[(df_obs['variable'] == 'doc') & (df_obs['depth'] == 0)]
df_obs_bot_doc= df_obs[(df_obs['variable'] == 'doc') & (df_obs['depth'] == 20)]

plt.figure(figsize=(10, 5))
plt.plot(times, doc_total[0,:]/volume[0], color='blue', label='0m Modeled DOC', linestyle='solid')
plt.plot(times, doc_total[40,:]/volume[40], color='blue', label='20m Modeled DOC', linestyle='dashed')
plt.scatter(df_obs_surf_doc["datetime"], df_obs_surf_doc["observation"], 
            color='red', label='0m Observed DOC', marker='o', zorder=5)
plt.scatter(df_obs_bot_doc["datetime"], df_obs_bot_doc["observation"], 
            color='red', label='20m Observed DOC', marker='x', zorder=5)
plt.ylabel("DOC (mg/L)", fontsize=15)
plt.xlabel("Time", fontsize=15)
plt.ylim(2, 8)
plt.legend(loc='best')
plt.show()

df_obs_temp=pd.read_csv('../input/observedTemp.txt',  parse_dates=['datetime'])
df_obs_temp['datetime'] = pd.to_datetime(df_obs_temp['datetime'], errors='coerce')
df_obs_temp = df_obs_temp[(df_obs_temp['datetime'] >= startingDate) & (df_obs_temp['datetime'] <= endingDate)]

df_obs_surf_temp = df_obs_temp[(df_obs_temp['Depth_meter'] == 1)]
df_obs_bot_temp = df_obs_temp[(df_obs_temp['Depth_meter'] == 22)]

plt.figure(figsize=(10, 5))
plt.plot(times, temp[2,:], color= 'blue', label='1m Modeled Temp', linestyle= 'solid')
plt.plot(times, temp[44,:], color= 'blue', label='22m Modeled Temp', linestyle= 'dashed')
plt.plot(df_obs_surf_temp["datetime"], df_obs_surf_temp["Water_Temperature_celsius"], color= 'red', label='1m Observed Temp', marker='o', zorder=5, linestyle= 'solid')
plt.plot(df_obs_bot_temp["datetime"], df_obs_bot_temp["Water_Temperature_celsius"], color= 'red', label='22m Observed Temp',  marker='o', zorder=5, linestyle= 'dashed')
plt.ylabel("Temp.", fontsize=15)
plt.xlabel("Time", fontsize=15) 
plt.legend(loc='best')
plt.show()
# TODO
# air water exchange
# sediment loss POC
# diffusive transport
# r and npp
# phosphorus bcDesktop/Research/Code/1D-AEMpy-UW-metabolism-EM/output/PB ME/
# ice npp
# wind mixingS
# poc_tot = np.add(pocl, pocr)
# pd.DataFrame(temp).to_csv("/Users/emmamarchisin/Desktop/Research/Code/1D-AEMpy-UW-metabolism-EM/output/PB ME/ME.modeled_temp.csv")
# pd.DataFrame(o2).to_csv("/Users/emmamarchisin/Desktop/Research/Code/1D-AEMpy-UW-metabolism-EM/output/PB ME/ME.modeled_do.csv")
# pd.DataFrame((o2/ volume[:, np.newaxis]).T).to_csv("/Users/emmamarchisin/Desktop/Research/Code/1D-AEMpy-UW-metabolism-EM/output/PB ME/ME.modeled_do_mgL.csv")
# pd.DataFrame(docr).to_csv("/Users/emmamarchisin/Desktop/Research/Code/1D-AEMpy-UW-metabolism-EM/output/PB MEME.modeled_docr.csv")
# pd.DataFrame(docl).to_csv("/Users/emmamarchisin/Desktop/Research/Code/1D-AEMpy-UW-metabolism-EM/output/PB ME/ME.modeled_docl.csv")
# pd.DataFrame(pocl).to_csv("/Users/emmamarchisin/Desktop/Research/Code/1D-AEMpy-UW-metabolism-EM/output/PB ME/ME.modeled_pocl.csv")
# pd.DataFrame((poc_tot/ volume[:, np.newaxis]).T).to_csv("/Users/emmamarchisin/Desktop/Research/Code/1D-AEMpy-UW-metabolism-EM/output/PB ME/ME.modeled_pocall_ugL.csv")
# pd.DataFrame(pocr).to_csv("/Users/emmamarchisin/Desktop/Research/Code/1D-AEMpy-UW-metabolism-EM/output/PB ME/ME.modeled_pocr.csv")
# pd.DataFrame(secchi).to_csv("/Users/emmamarchisin/Desktop/Research/Code/1D-AEMpy-UW-metabolism-EM/output/PB ME/ME.modeled_secchi.csv")
# pd.DataFrame(thermo_dep).to_csv("/Users/emmamarchisin/Desktop/Research/Code/1D-AEMpy-UW-metabolism-EM/output/PB ME/ME.modeled_thermo_dep.csv")
# pd.DataFrame(times).to_csv("/Users/emmamarchisin/Desktop/Research/Code/1D-AEMpy-UW-metabolism-EM/output/PB ME/ME.modeled_times.csv")


# pd.DataFrame(temp).to_csv("D:/bensd/Documents/RStudio Workspace/1D-AEM-py/model_output/modeled_temp.csv")
# pd.DataFrame(o2).to_csv("D:/bensd/Documents/RStudio Workspace/1D-AEM-py/model_output/modeled_do.csv")
# pd.DataFrame(docr).to_csv("D:/bensd/Documents/RStudio Workspace/1D-AEM-py/model_output/modeled_docr.csv")
# pd.DataFrame(docl).to_csv("D:/bensd/Documents/RStudio Workspace/1D-AEM-py/model_output/modeled_docl.csv")
# pd.DataFrame(pocl).to_csv("D:/bensd/Documents/RStudio Workspace/1D-AEM-py/model_output/modeled_pocl.csv")
# pd.DataFrame(pocr).to_csv("D:/bensd/Documents/RStudio Workspace/1D-AEM-py/model_output/modeled_pocr.csv")
# pd.DataFrame(secchi).to_csv("D:/bensd/Documents/RStudio Workspace/1D-AEM-py/model_output/modeled_secchi.csv")


# label = 116
# doc_tot = np.add(docl, docr)
# poc_tot = np.add(pocl, pocr)
# os.mkdir("../parameterization/output/Run_"+str(label))
# pd.DataFrame(temp).to_csv("../parameterization/output/Run_"+str(label)+"/temp.csv", index = False)
# pd.DataFrame(o2).to_csv("../parameterization/output/Run_"+str(label)+"/do.csv", index = False)
# pd.DataFrame(doc_all).to_csv("../parameterization/output/Run_"+str(label)+"/doc.csv", index = False)
# pd.DataFrame(poc_all).to_csv("../parameterization/output/Run_"+str(label)+"/poc.csv", index = False)
# pd.DataFrame(secchi).to_csv("../parameterization/output/Run_"+str(label)+"/secchi.csv", index = False)


# test clustering script
# tendencies?
dO2_dt = np.gradient(o2, dt, axis = 1)

# gradients
dO2_dz = np.gradient(o2, dx, axis = 0)
dT_dz = np.gradient(temp, dx, axis = 0)

# boundary states
ice_bc = np.tile(icethickness, (len(depth), 1))
clarity_bc = np.tile(secchi, (len(depth), 1))

# feature matrix
mask = np.isfinite(o2)
X = np.column_stack([
    o2.ravel(),
    dO2_dt.ravel(),
    dO2_dz.ravel(),
    temp.ravel(),
    dT_dz.ravel(),
    ice_bc.ravel(),
    clarity_bc.ravel()
])

#breakpoint()

X = X[mask.ravel()]

from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

X_scaled[:, 0] *= 2.0  # weight DO more heavily
X_scaled[:, 1] *= 2.0  # weight gradient heavily

# find regimes
import hdbscan 

cluster = hdbscan.HDBSCAN(min_cluster_size=4, min_samples=20, metric = 'euclidean')

labels = cluster.fit_predict(X_scaled)

cluster_map = np.full(o2.shape, np.nan)

labels = labels.reshape(len(depth), -1)
cluster_map = labels
cluster_map = cluster_map.reshape(o2.shape)



fig, ax = plt.subplots(figsize=(15,5))
sns.heatmap(cluster_map , cmap=plt.cm.get_cmap('Spectral_r'),  xticklabels=1000, yticklabels=2)
ax.contour(np.arange(.5, temp.shape[1]), np.arange(.5, temp.shape[0]), calc_dens(temp), levels=[999],
           colors=['black', 'gray'],
           linestyles = 'dotted')
ax.set_ylabel("Depth (m)", fontsize=15)
ax.set_xlabel("Time", fontsize=15)    
ax.collections[0].colorbar.set_label("Cluster")
xticks_ix = np.array(ax.get_xticks()).astype(int)
time_label = times[xticks_ix]
nelement = len(times)//N_pts
#time_label = time_label[::nelement]
#ax.xaxis.set_major_locator(plt.MaxNLocator(N_pts * n_years))
ax.set_xticklabels(time_label.strftime("%d-%m-%y"), rotation=45, ha = 'right')
yticks_ix = np.array(ax.get_yticks()).astype(int)
depth_label = yticks_ix / 2
ax.set_yticklabels(depth_label, rotation=0)
plt.show()