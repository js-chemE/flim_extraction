import configparser
import warnings

from flim_extraction import process_folder

CONFIG = configparser.ConfigParser()
CONFIG.read("path to config")

if __name__ == "__main__":
    warnings.resetwarnings()
    warnings.simplefilter("ignore", RuntimeWarning)
    # print(read_calibration(CONFIG.get("VALUE"", "calibration_path")))
    calibration_kwargs = {}
    calibration_kwargs["calibration_path"] = CONFIG.get("VALUE", "calibration_path")
    calibration_kwargs["tau1"] = CONFIG.getfloat("FUNCTION", "tau1")
    calibration_kwargs["tau2"] = CONFIG.getfloat("FUNCTION", "tau2")
    calibration_kwargs["pka1"] = CONFIG.getfloat("FUNCTION", "pka1")
    calibration_kwargs["pka2"] = CONFIG.getfloat("FUNCTION", "pka2")
    calibration_kwargs["N"] = CONFIG.getint("FUNCTION", "N")
    calibration_kwargs["pH_range"] = (
        CONFIG.getfloat("FUNCTION", "ph_min"),
        CONFIG.getfloat("FUNCTION", "ph_max"),
    )
    print(calibration_kwargs)
    process_folder(
        CONFIG.get("SAMPLES", "folder"),
        CONFIG.get("REFERENCE", "path"),
        CONFIG.getfloat("REFERENCE", "lifetime"),
        CONFIG.getfloat("REFERENCE", "frequency"),
        CONFIG.get("VALUE", "calibration_method"),
        calibration_kwargs,
        filter_in=CONFIG.get("FILTER", "in").split(", "),
        filter_out=CONFIG.get("FILTER", "out").split(", "),
        max_files=CONFIG.getint("SAMPLES", "max_files"),
        max_frames=CONFIG.getint("SAMPLES", "max_frames"),
    )
    warnings.resetwarnings()
