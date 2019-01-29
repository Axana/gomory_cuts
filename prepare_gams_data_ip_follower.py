"""
Preparing wind, solar, and electrical load data
for GAMS computations for a given time discretization.
"""

# Global imports
import argparse
import pandas
import numpy
import json
import glob
import os
import datetime
import sys


class GamsDataPreparateur(object):
    """
    Class for preparing all GAMS input data.
    """

    def __init__(self, arguments):
        """C'tor."""
        self.arguments = arguments

    def prepare_data(self):
        """Main method for preparing the GAMS data."""

        # parse data
        dateparse = lambda x: pandas.datetime.strptime(x, "%d.%m.%Y %H:%M")
        dataframe = pandas.read_csv(self.arguments["weather_data"], sep="\t", na_values="-", parse_dates=True,
                                    date_parser=dateparse, keep_date_col=True, header=0, index_col=0)
        dataframe.columns.names = ["TimeStamp"]
        time_start = datetime.date(int(self.arguments["year"]), int(self.arguments["month"]), int(self.arguments["day"]))
        time_end = time_start + datetime.timedelta(hours=25)
        converted = dataframe.truncate(before=time_start, after=time_end).asfreq(self.arguments["time_step"], method=None)
        dataframe_idx = converted.interpolate(method="linear", axis=0,
                                              inplace=False, downcast=None, limit=None)
        dataframe_idx = dataframe_idx[time_start.isoformat()]
        # dataframe_idx = dataframe_idx.resample(self.arguments["time_step"])
        
        # Time discretization parameters calculation
        discretization_frequency = pandas.to_timedelta(self.arguments["time_step"])
        Delta_t = discretization_frequency.total_seconds()/3600
        number_of_time_periods_in_a_day = dataframe_idx.groupby(dataframe_idx.index.date).size()[0]
        number_of_days = len(dataframe_idx.groupby(dataframe_idx.index.date).size().index)
        number_of_time_periods = number_of_days * number_of_time_periods_in_a_day
        print(str(discretization_frequency) + "; " + str(Delta_t) + "; "  + str(number_of_days) + "; "  + str(number_of_time_periods_in_a_day) + "; "  + str(number_of_time_periods))
        
        # ... and writing for GAMS
        time_discretization_dat_filename = os.path.join("./gams_input_data/", "time_discretization.dat")
        time_discretization = open(time_discretization_dat_filename, "w")
        time_discretization.write("scalar Delta_t Time step /" + str(Delta_t) + "/ ; \n")
        time_discretization.write("scalar number_of_time_periods Overall number of time periods in the considered time horizont /" 
                                + str(number_of_time_periods) + "/ ; \n")
        time_discretization.write("scalar number_of_days Number of days in the considered time horizont /" + str(number_of_days) + "/ ; \n")
        time_discretization.write("scalar number_of_time_periods_in_a_day Number of time periods in one day /" + str(number_of_time_periods_in_a_day) + "/ ; \n")
        time_discretization.close()
        
        # CPP time start and end
        cpp_points_dat_filename = os.path.join("./gams_input_data/", "cpp_points.dat")
        cpp_points = open(cpp_points_dat_filename, "w")
        cpp_time_start_1 = pandas.to_timedelta(self.arguments["cpp_time_start_1"]).total_seconds()//3600/Delta_t
        cpp_time_end_1 = pandas.to_timedelta(self.arguments["cpp_time_end_1"]).total_seconds()//3600/Delta_t
        print("CPP time start 1 = " + str(cpp_time_start_1))
        print("CPP time end 1 = " + str(cpp_time_end_1))
        # pandas.to_datetime(self.arguments["cpp_time_start"], format='%H:%M')
        if cpp_time_start_1.is_integer(): 
            cpp_points.write("scalar cpp_time_start_1 Critical period 1 first hour /" + str(int(cpp_time_start_1)) + "/ ; \n")
        else: sys.exit("Specified CPP starting time point 1 is not a valid point of the discretized time horizon!")
        if cpp_time_end_1.is_integer(): 
            cpp_points.write("scalar cpp_time_end_1 Critical period 1 last hour /" + str(int(cpp_time_end_1)) + "/ ; \n")
        else: sys.exit("Specified CPP end time point 1 is not a valid point of the discretized time horizon!")
        cpp_time_start_2 = pandas.to_timedelta(self.arguments["cpp_time_start_2"]).total_seconds()//3600/Delta_t
        cpp_time_end_2 = pandas.to_timedelta(self.arguments["cpp_time_end_2"]).total_seconds()//3600/Delta_t
        print("CPP time start 2 = " + str(cpp_time_start_2))
        print("CPP time end 2 = ", cpp_time_end_2)
        # pandas.to_datetime(self.arguments["cpp_time_start"], format='%H:%M')
        if cpp_time_start_2.is_integer(): 
            cpp_points.write("scalar cpp_time_start_2 Critical period 2 first hour /" + str(int(cpp_time_start_2)) + "/ ; \n")
        else: sys.exit("Specified CPP starting time point 2 is not a valid point of the discretized time horizon!")
        if cpp_time_end_2.is_integer(): 
            cpp_points.write("scalar cpp_time_end_2 Critical period 2 last hour /" + str(int(cpp_time_end_2)) + "/ ; \n")
        else: sys.exit("Specified CPP end time point 2 is not a valid point of the discretized time horizon!")
        cpp_points.close()


        dataframe_idx.index = numpy.arange(0, len(dataframe_idx))

        # writing GAMS compatible wind data file
        wind_csv_filename = "wind.csv"
        wind_dat_filename = os.path.join(self.arguments["gams_input_data_dir"], wind_csv_filename.replace(".csv", ".dat"))
        dataframe_idx["Wind"].to_csv(wind_csv_filename, sep="\t", header=None)
        wind_file_header = "parameters V_wind_s(Gamma) Wind velocity in Nuernberg \n/ \n"
        open(wind_dat_filename, "w").write(wind_file_header + open(wind_csv_filename).read())
        with open(wind_dat_filename, "a", encoding="utf-8") as wind_dat_file:
            wind_dat_file.write("/")

        # writing GAMS compatible solar file
         # convert Wh to kWh
        dataframe_idx["Pac"] = dataframe_idx["Pac"]/1000.0
        pv_csv_filename = "pv.csv"
        pv_dat_filename = os.path.join(self.arguments["gams_input_data_dir"], pv_csv_filename.replace(".csv", ".dat"))
        dataframe_idx["Pac"].to_csv(pv_csv_filename, sep="\t", header=None)
        pv_file_header = "parameters P_pv_s(Gamma) Power generated by PV in kWh \n/ \n"
        open(pv_dat_filename, "w").write(pv_file_header + open(pv_csv_filename).read())
        with open(pv_dat_filename, "a", encoding="utf-8") as pv_dat_file:
            pv_dat_file.write("/")

        # writing temperature profile of the given day
        # not(!) required for the computations, but maybe
        # useful for plots etc.
        temp_dat_filename = os.path.join(self.arguments["gams_input_data_dir"], "temp.dat")
        dataframe_idx["Ta"].to_csv(temp_dat_filename, sep="\t", header=None)

        # preparing electricity load data for GAMS
        dateparseElec = lambda x: pandas.datetime.strptime(x, "%m/%d/%Y %H:%M:%S")
        elec = pandas.read_csv(self.arguments["electrical_load"], sep="\t", na_values="-",
                               parse_dates=True, date_parser=dateparseElec, keep_date_col=True, header=0, index_col=0)
        elec_converted = elec.truncate(before=time_start, after=time_end).asfreq(self.arguments["time_step"], method=None)
        elec_idx = elec_converted.interpolate(method="linear", axis=0,
                                              inplace=False, downcast=None, limit=None)              
        elec_idx = elec_idx[time_start.isoformat()]
        elec_idx = elec_idx.resample(self.arguments["time_step"])
        elec_idx.index = numpy.arange(0, len(elec_idx))
        # demand is given in 1e6 kWh
        # SCALING to kWh/10 and INTEGER
        elec_idx_scaled = (elec_idx/float(1e2)).round(0)
        #elec_idx_scaled = elec_idx/float(1e6)
        # writing GAMS compatible electrical load data file
        elec_csv_filename = "elec_load.csv"
        elec_dat_filename = os.path.join(self.arguments["gams_input_data_dir"], elec_csv_filename.replace(".csv", ".dat"))
        elec_idx_scaled.to_csv(elec_csv_filename, sep="\t", header=None)
        elec_load_file_header = "parameters P_le_total_orig_s(Gamma) Total electrical load of the house \n/ \n"
        open(elec_dat_filename, "w").write(elec_load_file_header + open(elec_csv_filename).read())
        with open(elec_dat_filename, "a", encoding="utf-8") as elec_dat_file:
            elec_dat_file.write("/")

        # prepare heat load data for GAMS
        temp_mean = dataframe_idx.mean()["Ta"]
        heat_load = pandas.read_csv(self.arguments["load_gas_intraday"], sep="\t", na_values="-",
                                   parse_dates=True, keep_date_col=True, index_col=0)
        daily_fraction_of_heat_load = pandas.read_csv(self.arguments["load_gas_per_day"], sep="\t", na_values="-",
                                   index_col=0)
        discretized_heat_load = heat_load.asfreq(self.arguments["time_step"], method=None)
        discretized_heat_load_idx = discretized_heat_load.interpolate(method="linear", axis=0, inplace=False, downcast=None, limit=None)
        discretized_heat_load_idx.index = numpy.arange(0, len(discretized_heat_load_idx))
        discretized_heat_load_idx = discretized_heat_load_idx[0:len(discretized_heat_load_idx)-1]
        discretized_heat_load_idx = discretized_heat_load_idx * daily_fraction_of_heat_load.loc[int(self.arguments["month"]), "Daily heat load as a fraction of yearly load, dependent on the month"]
        for temp_class in list(discretized_heat_load_idx.columns.values):
           if float(temp_class) < temp_mean:
               continue
           else:
               # writing GAMS compatible heat load data file
               heat_load_csv_filename = "heat_load.csv"
               heat_load_dat_filename = os.path.join(self.arguments["gams_input_data_dir"], heat_load_csv_filename.replace(".csv", ".dat"))
               discretized_heat_load_idx[temp_class].to_csv(heat_load_csv_filename, sep="\t", header=None)
               heat_load_file_header = "parameters P_lh_intraday_s(Gamma) Fractions of daily gas load of the house \n/ \n"
               open(heat_load_dat_filename, "w").write(heat_load_file_header + open(heat_load_csv_filename).read())
               with open(heat_load_dat_filename, "a", encoding="utf-8") as heat_load_dat_file:
                   heat_load_dat_file.write("/")
               break

        # prepare EEX spot price data for GAMS
        eex_price = pandas.read_csv(self.arguments["eex_spot_price"], sep="\t", na_values="-", parse_dates=True,
                                   date_parser=dateparse, keep_date_col=True, header=0, index_col=0)
        discretized_eex_price = eex_price.truncate(before=time_start, after=time_end).asfreq(self.arguments["time_step"], method=None)
        
        discretized_eex_price_idx = discretized_eex_price.interpolate(method="linear", axis=0, inplace=False, downcast=None, limit=None)
        discretized_eex_price_idx.index = numpy.arange(0, len(discretized_eex_price_idx))
        discretized_eex_price_idx = discretized_eex_price_idx[0:len(discretized_eex_price_idx)-1]       
        # SCALING to EUR per kWh/10, NOT INTEGER
        eex_price_idx_scaled = discretized_eex_price_idx / 10000
        #eex_price_idx_scaled = discretized_eex_price_idx
        eex_price_csv_filename = "eex_spot_price.csv"
        eex_price_dat_filename = os.path.join(self.arguments["gams_input_data_dir"], eex_price_csv_filename.replace(".csv", ".dat"))
        eex_price_idx_scaled.to_csv(eex_price_csv_filename, sep="\t", header=None)
        eex_price_load_file_header = "parameters gamma_eex(Gamma) EEX spot price in EUR for MWh \n/ \n"
        open(eex_price_dat_filename, "w").write(eex_price_load_file_header + open(eex_price_csv_filename).read())
        with open(eex_price_dat_filename, "a", encoding="utf-8") as eex_price_dat_file:
            eex_price_dat_file.write("/")

        # remove intermediate csv files
        for csv_filename in glob.glob("./*.csv"):
            os.remove(csv_filename)

        # remove intermediate csv files
        for csv_filename in glob.glob("./*.csv"):
            os.remove(csv_filename)

        # Number of time intervals for GAMS file
        return ([(number_of_time_periods-1), (number_of_time_periods_in_a_day-1)])

def main():
    """Main function."""

    # parsing arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument("ctrl_file")
    console_arguments = argparser.parse_args()
    arguments = json.loads(open(console_arguments.ctrl_file).read())

    gams_data_preparateur = GamsDataPreparateur(arguments)
    gams_data_preparateur.prepare_data()


if __name__ == "__main__":
    main()
