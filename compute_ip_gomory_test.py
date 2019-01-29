"""
Smart Grid Optimization.
"""

# Global imports
import subprocess
import argparse
import logging
import shutil
import json
import glob
import sys
import os
import pandas
import math
import numpy as np
import time

start_time=time.clock()
sys.setrecursionlimit(2000)

# Local imports
from prepare_gams_data_ip_follower import GamsDataPreparateur

# init logger
LOGGER = logging.getLogger(__name__)

cutcounter=0
# files for duality conditions and cutting planes
vardef_file=open("gom_dual_variables_definition.dat", "w")  # defining dual variables for cutting plane
equationdef_file=open("gom_equations_definition.dat", "w")  # defining cutting plane
parameter_file=open("gom_cut_parameters.dat", "w")          # parameters for cutting plane
cut_file=open("gom_cutting_planes.dat", "w")                # specify cutting plane
dualfeas_file=open("gom_dual_feasibility.dat", "w")         # dual feasibility conditions
strongdual_file=open("gom_strong_duality.dat", "w")         # strong duality condition
gams_calculation_file=open("calculate_parameters.dat","w")   # Gomory parameters to calculate by GAMS

# basic strong duality
strong_duality="strong_duality\t.. obj_ir_lower =E= sum(Gamma, l_z_on_up_bd(Gamma) + l_z_up_up_bd(Gamma) + H_max * l_H_up_bd(Gamma) - H_min * l_H_low_bd(Gamma) + E_max * l_E_up_bd(Gamma) + P_boiler_max * Delta_t * l_boiler_up_bd(Gamma) +P_bat_c_max * Delta_t * l_bat_c_up_bd(Gamma) + P_bat_d_max * Delta_t * l_bat_d_up_bd(Gamma) + P_le_total(Gamma) * (l_elec_bal_l(Gamma) - l_elec_bal_g(Gamma)) - P_lh_total(Gamma-1)$(ord(Gamma) > 1) / eta_hsu_d * (l_hsu_bal_l(Gamma)$(ord(Gamma) > 1) - l_hsu_bal_g(Gamma)$(ord(Gamma) > 1)) - (H_start + P_lh_total(Gamma) / eta_hsu_d)$(ord(Gamma) = card(Gamma)) * l_hsu_terminal_lower + (H_max + P_lh_total(Gamma) / eta_hsu_d)$(ord(Gamma) = card(Gamma)) * l_hsu_terminal_upper) + H_start * (l_init_hsu_l - l_init_hsu_g) + E_start * (l_init_bat_l - l_init_bat_g) - E_start * l_bat_terminal_lower + E_max * l_bat_terminal_upper;"
strongdual_file.write(strong_duality)

# basic dual feasibility
dual_feasibility="dual_P_im(Gamma)\t.. l_elec_bal_l(Gamma) - l_elec_bal_g(Gamma) =G= - (gamma_tax + gamma_ret + gamma_cpp(Gamma));\ndual_P_boiler(Gamma)\t.. l_boiler_up_bd(Gamma) - (l_elec_bal_l(Gamma) - l_elec_bal_g(Gamma)) - (l_hsu_input_l(Gamma) - l_hsu_input_g(Gamma)) =G= 0;\ndual_P_chp_ex(Gamma)\t.. l_chp_ex_upper(Gamma) - (l_elec_bal_l(Gamma) - l_elec_bal_g(Gamma)) =G= gamma_chp_fit;\ndual_P_chp_e(Gamma)\t.. (l_elec_bal_l(Gamma) - l_elec_bal_g(Gamma)) + l_chp_e_upper(Gamma) - l_chp_e_lower(Gamma) - zeta * l_chp_h_upper(Gamma) - l_chp_ex_upper(Gamma) =G= gamma_chp_sub - gamma_gas * (1 + zeta) / eta_chp;\ndual_P_chp_h(Gamma)\t.. l_chp_h_upper(Gamma) - (l_hsu_input_l(Gamma) - l_hsu_input_g(Gamma)) =G= 0;\ndual_P_hsu_c(Gamma)\t.. (l_hsu_input_l(Gamma) - l_hsu_input_g(Gamma)) - eta_hsu_c * (l_hsu_bal_l(Gamma + 1)$(ord(Gamma) < card(Gamma)) - l_hsu_bal_g(Gamma + 1)$(ord(Gamma) < card(Gamma))) +  eta_hsu_c * (l_hsu_terminal_upper - l_hsu_terminal_lower)$(ord(Gamma) = card(Gamma))  =G= 0;\ndual_H(Gamma)\t.. l_H_up_bd(Gamma) - l_H_low_bd(Gamma) + (l_init_hsu_l$(ord(Gamma) = 1) - l_init_hsu_g$(ord(Gamma) = 1)) + (l_hsu_bal_l(Gamma)$(ord(Gamma) > 1) - l_hsu_bal_g(Gamma)$(ord(Gamma) > 1)) - (1-alpha_hsu) * (l_hsu_bal_l(Gamma + 1)$(ord(Gamma) < card(Gamma)) - l_hsu_bal_g(Gamma + 1)$(ord(Gamma) < card(Gamma))) + (1-alpha_hsu) * (l_hsu_terminal_upper - l_hsu_terminal_lower)$(ord(Gamma) = card(Gamma)) =G= 0;\ndual_E(Gamma)\t.. l_E_up_bd(Gamma) + (l_init_bat_l$(ord(Gamma) = 1) - l_init_bat_g$(ord(Gamma) = 1)) + (l_bat_bal_l(Gamma)$(ord(Gamma) > 1) - l_bat_bal_g(Gamma)$(ord(Gamma) > 1)) - (1-alpha_bat) * (l_bat_bal_l(Gamma + 1)$(ord(Gamma) < card(Gamma)) - l_bat_bal_g(Gamma + 1)$(ord(Gamma) < card(Gamma))) + (1-alpha_bat) * (l_bat_terminal_upper - l_bat_terminal_lower)$(ord(Gamma) = card(Gamma)) =G= 0;\ndual_P_bat_c(Gamma)\t.. l_bat_c_up_bd(Gamma) - (l_elec_bal_l(Gamma) - l_elec_bal_g(Gamma)) - eta_bat_c * (l_bat_bal_l(Gamma + 1)$(ord(Gamma) < card(Gamma)) - l_bat_bal_g(Gamma + 1)$(ord(Gamma) < card(Gamma))) + eta_bat_c * (l_bat_terminal_upper - l_bat_terminal_lower)$(ord(Gamma) = card(Gamma))  =G= 0;\ndual_P_bat_d(Gamma)\t.. l_bat_d_up_bd(Gamma) + (l_elec_bal_l(Gamma) - l_elec_bal_g(Gamma)) + (l_bat_bal_l(Gamma + 1)$(ord(Gamma) < card(Gamma)) - l_bat_bal_g(Gamma + 1)$(ord(Gamma) < card(Gamma))) / eta_bat_d + (l_bat_terminal_lower - l_bat_terminal_upper)$(ord(Gamma) = card(Gamma)) / eta_bat_d  =G= 0;\ndual_z_on(Gamma)\t.. l_z_on_up_bd(Gamma) + l_z_up_lower(Gamma) - l_z_up_lower(Gamma + 1)$(ord(Gamma) < card(Gamma)) - P_chp_e_max * Delta_t * l_chp_e_upper(Gamma) + k * P_chp_e_max * Delta_t * l_chp_e_lower(Gamma) =G= 0;\ndual_z_up(Gamma)\t.. l_z_up_up_bd(Gamma) - l_z_up_lower(Gamma) =G= - gamma_gas * g_chp;"
dualfeas_file.write(dual_feasibility)

vardef_file.close()
equationdef_file.close()
parameter_file.close()
cut_file.close()
dualfeas_file.close()
strongdual_file.close()
gams_calculation_file.close()

def main():
    """
    Main method.

    Coordinates the preparation of
    - producing the GAMS input data,
    - the solution of the GAMS model, and
    - cleaning up the directories.
    """


    
    # configure verbosity level of logger
    logging.basicConfig(level=logging.INFO)

    # parsing arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument("ctrl_file")
    console_arguments = argparser.parse_args()
    arguments = json.loads(open(console_arguments.ctrl_file).read())

    # prepare result directory
    result_subdir_name = arguments["results_dir_prefix"] + arguments["year"] + "." + arguments["month"] + "." + arguments["day"] + "_" + arguments["time_step"]
    full_result_dir_path = os.path.join(arguments["results_base_dir"], result_subdir_name)
    if os.path.isdir(full_result_dir_path):
        LOGGER.info("Result directory " + full_result_dir_path + " already exists. Stopping everything.")
        sys.exit()
    else:
        os.mkdir(full_result_dir_path)

        # prepare GAMS data
    gams_data_preparateur = GamsDataPreparateur(arguments)
    nr_time_intervals = gams_data_preparateur.prepare_data()
    
    def run_gams():
        print("Initiate GAMS model.")
        # run GAMS prosumer model
        gams_log_filename = arguments["gams_input_file_prosumer"] + ".log"
        # nr_time_intervals[0] = int(24 * 60 / 15.) - 1
        if arguments["logging"] == "yes":
            with open(gams_log_filename, "w") as gams_log_file:
                gams_process = subprocess.Popen(["gams", (arguments["gams_input_file_prosumer"] + ".gms"), "lo=3",
                                            "u1=" + str(nr_time_intervals[0]), "u2=" + str(nr_time_intervals[1])],
                                            stdout=gams_log_file,
                                            stderr=subprocess.STDOUT)
                gams_process.communicate()
        else:
            gams_process = subprocess.Popen(["gams", (arguments["gams_input_file_prosumer"] + ".gms"), "lo=3",
                                            "u1=" + str(nr_time_intervals[0]), "u2=" + str(nr_time_intervals[1])])
            gams_process.communicate()

        # run GAMS bilevel model
        gams_log_filename = arguments["gams_input_file_bilevel"] + ".log"
        # nr_time_intervals[0] = int(24 * 60 / 15.) - 1
        if arguments["logging"] == "yes":
            with open(gams_log_filename, "w") as gams_log_file:
                gams_process = subprocess.Popen(["gams", (arguments["gams_input_file_bilevel"] + ".gms"), "lo=3",
                                            "u1=" + str(nr_time_intervals[0]), "u2=" + str(nr_time_intervals[1])],
                                            stdout=gams_log_file,
                                            stderr=subprocess.STDOUT)
                gams_process.communicate()
        else:
            gams_process = subprocess.Popen(["gams", (arguments["gams_input_file_bilevel"] + ".gms"), "lo=3",
                                            "u1=" + str(nr_time_intervals[0]), "u2=" + str(nr_time_intervals[1])])
            gams_process.communicate()    
            # run GAMS lower level with values for upper level variables taken from the bilevel solution
        gams_log_filename = arguments["gams_input_file_feas_test"] + ".log"
        # nr_time_intervals[0] = int(24 * 60 / 15.) - 1
        if arguments["logging"] == "yes":
            with open(gams_log_filename, "w") as gams_log_file:
                gams_process = subprocess.Popen(["gams", (arguments["gams_input_file_feas_test"] + ".gms"), "lo=3",
                                            "u1=" + str(nr_time_intervals[0]), "u2=" + str(nr_time_intervals[1])],
                                            stdout=gams_log_file,
                                            stderr=subprocess.STDOUT)
                gams_process.communicate()
        else:
            gams_process = subprocess.Popen(["gams", (arguments["gams_input_file_feas_test"] + ".gms"), "lo=3",
                                            "u1=" + str(nr_time_intervals[0]), "u2=" + str(nr_time_intervals[1])])
            gams_process.communicate()
        return
 
    # relevant variables linked to matrix columns 
    colvar=['P_im', 'P_chp_h', 'P_chp_e', 'z_on', 'z_up', 'P_bat_c', 'P_bat_d', 'E' , 'P_boiler', 'H', 'P_chp_ex', 'P_hsu_c' ]

        
    def restart():# read GAMS results
        run_gams()
        data_name="ip_follower_IP_leader_"+ arguments["tariff_type"]+"_M0.csv"
        gms_results = pandas.read_csv(data_name, 
                        sep=";", na_values="-", keep_date_col=True, header=0, index_col=0, encoding='utf-8-sig')
        return int_check(gms_results)    
    
    # verify integrality
    def int_check(results):
        results.index.name = ["Index"]
        gams_solution=[]                     #read variables for later cutting plane test
        for variablename in colvar:
            gams_solution.extend(results[variablename])
        gams_solution.extend([-1])
        vector = pandas.concat([results['z_on'],results['z_up']])
        for y in vector:
            if y.is_integer()==False:
                global cutcounter
                cutcounter+=1
                print("Apply gomory cut", cutcounter)
                return create_gomorycut(gams_solution)
        print("Integer valued optimal solution found. Number of applied gomory cuts: ",cutcounter)
        
        return 
    
    # Gomory cutting plane method
    def create_gomorycut(solution):
        
        vardef_file=open("gom_dual_variables_definition.dat", "a")  # defining dual variables for cutting plane
        equationdef_file=open("gom_equations_definition.dat", "a")  # defining cutting plane
        parameter_file=open("gom_cut_parameters.dat", "a")          # parameters for cutting plane
        cut_file=open("gom_cutting_planes.dat", "a")                # specify cutting plane
        
        
        # read_gms_data():
        gomory_cut_parameter_file="gomory_parameters.csv"
        gom_par=pandas.read_csv(gomory_cut_parameter_file, delimiter=";", index_col="Variable")
        
        vector=np.asarray(gom_par["gomory_cut_value"])
        gomory_vector=np.floor(vector)# gomory parameters as vector for cut test
        
        vardef_file.write("\n positive variables \t l_gomory_cut_" + str(cutcounter) + "\t dual variable for gomory cut; \n")
        equationdef_file.write("equations \t gomory_cut_" + str(cutcounter) + ";\n")
        cut_file.write("gomory_cut_" + str(cutcounter) + " .. sum(Gamma, " )
        variable_type=""
        
        for variable in gom_par.index:
            if variable!="right_side":
                if variable_type != variable.split("(")[0]:
                    variable_type = variable.split("(")[0]
                    
                    parameter_file.write("parameters \t gomory" + str(cutcounter)+ "_" + variable_type + "(Gamma) \t parameters of gomory cuts; \n")
                    cut_file.write(" + gomory" + str(cutcounter) + "_" + variable_type + "(Gamma) * " + variable_type + "(Gamma)" )
            else:
                parameter_file.write("parameter \t gomory" + str(cutcounter)+ "_right_side \t right side of gomory cut; \n")
                cut_file.write(") =L= gomory" + str(cutcounter) + "_right_side;\n")
            gomory_parameter_name = "gomory"  + str(cutcounter)+ "_" + variable
            gomory_value=np.floor(gom_par["gomory_cut_value"][variable])
            parameter_file.write(gomory_parameter_name + " = " +str(gomory_value) + ";\n")
            
        cuttest(gomory_vector,solution)                                    # test if cutting plane works
        
        rewrite_dual_feasibility(cutcounter)
        rewrite_strong_duality(cutcounter)
        rewrite_parameter_calculation(cutcounter)
        
        vardef_file.close()
        equationdef_file.close()
        parameter_file.close()
        cut_file.close()
        
        save_result_dir=os.path.join(full_result_dir_path, ("result_gomory_cut" + str(cutcounter-1) + "/"))
        os.mkdir(save_result_dir)
        shutil.move(("values_of_dual_variables_for_bounds.dat"), save_result_dir)
        shutil.move((arguments["gams_input_file_prosumer"] + ".lst"), save_result_dir)
        shutil.move((arguments["gams_input_file_bilevel"] + ".lst"), save_result_dir)
        shutil.move((arguments["gams_input_file_feas_test"] + ".lst"), save_result_dir)
        shutil.move((arguments["gams_input_file_prosumer"] + ".dat"), save_result_dir)
        shutil.move((arguments["gams_input_file_bilevel"] + ".dat"), save_result_dir)
        shutil.move((arguments["gams_input_file_feas_test"] + ".dat"), save_result_dir)
        shutil.move((arguments["gams_input_file_prosumer"] + ".csv"), save_result_dir)
        shutil.move((arguments["gams_input_file_bilevel"] + ".csv"), save_result_dir)
        shutil.move((arguments["gams_input_file_feas_test"] + ".csv"), save_result_dir)
    
        restart()
        return
    
    def cuttest(gom_vector,solution):
        par_sol= np.array(solution,dtype=float)
        #print(gom_vector)
        #print(par_sol)
        multiply=np.inner(gom_vector,par_sol)
        if multiply > 0: 
            print("difference: ",multiply)
        else: 
            print("cut did not apply correctly! ", multiply)
    
    def rewrite_dual_feasibility(number):
        dfstr=""
        calculationfile=pandas.read_csv("gom_dual_feasibility.dat", delimiter="\n", header=None)
        for line in calculationfile[0]:
            variable=line.split("(Gamma)")[0]
            variable=variable.replace("dual_","")
            newline=line[:-1] + " - gomory"  + str(cutcounter)+ "_" + variable + "(Gamma) * l_gomory_cut_" + str(number) + ";\n"
            dfstr+=newline
        with open("gom_dual_feasibility.dat","w") as rewritedf:
            rewritedf.write(dfstr)
        return
    
    def rewrite_strong_duality(number):
        with open("gom_strong_duality.dat","r") as data:
            line=data.read()
            newline=line[:-1]+ " + gomory" + str(cutcounter) + "_right_side * l_gomory_cut_" + str(number) + ";"
        with open("gom_strong_duality.dat","w") as rewritesd:
            rewritesd.write(newline)
        return

    def rewrite_parameter_calculation(number):
        with open("calculate_parameters.dat","a") as pfile:
            for var in colvar:
                pfile.write("put l_gomory_cut_" + str(cutcounter) + ",@20,l_gomory_cut_" + str(cutcounter) + ".l /; \nloop(Gamma,  gomory_" + var + "(Gamma) = gomory_" + var + "(Gamma) + gomory" + str(cutcounter) + "_" + var + "(Gamma) * l_gomory_cut_" + str(cutcounter) + ".l);\n" )
        return
        
    restart()
    
    exe_time=(time.clock()-start_time)*60
    print("executing time [sec]: ",exe_time)
    with open("statistics/statistics.csv","a") as stats:
        stats.write( "\n" + arguments["year"]+"-"+arguments["month"]+"-"+arguments["day"]+ ";" +arguments["time_step"] + ";" + str(cutcounter) + ";" + str(exe_time))
    
    # Moving results files
    result_filenames = glob.glob("./*.eps") + glob.glob("./*.pdf") + glob.glob("./*.dat") + glob.glob("./*.log") + glob.glob("./*.csv")
    for result_filename in result_filenames:
        shutil.move(result_filename, full_result_dir_path)

    gams_input_data_dir_name = "gams_input_data"
    for gams_input_filename in glob.glob(os.path.join(gams_input_data_dir_name, "*.dat")):
        shutil.copy(gams_input_filename, full_result_dir_path)
    for gams_input_filename in glob.glob(os.path.join(gams_input_data_dir_name, "*.csv")):
        shutil.copy(gams_input_filename, full_result_dir_path)
        
    shutil.move((arguments["gams_input_file_prosumer"] + ".lst"), full_result_dir_path)
    shutil.move((arguments["gams_input_file_bilevel"] + ".lst"), full_result_dir_path)
    shutil.move((arguments["gams_input_file_feas_test"] + ".lst"), full_result_dir_path)
if __name__ == "__main__":
    main()


# check binary variables +++ Gomory cutting plane method


#gmssol=full_result_dir_path + 'smart_grid_M0_SD_P_im_Beta_Binary.dat'

