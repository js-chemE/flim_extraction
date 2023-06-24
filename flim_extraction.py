import json
import os
from typing import List, Tuple, Union

import fdflim as fd
import flifile as ff
import numpy as np


def get_data(
    sample_path: str,
    reference_path: str,
    reference_lifetime: float,
    reference_frequency: float,
    max_frames: int = 1000,
) -> Tuple[List[fd.Sample], dict]:
    sample_raw = ff.FliFile(sample_path)
    reference_raw = ff.FliFile(reference_path)
    reference = fd.Reference(
        reference_raw.getdata(), reference_lifetime, reference_frequency
    )
    data = sample_raw.getdata()
    samples = []
    if data.ndim <= 3:
        samples.append(fd.Sample(data, reference))
    else:
        number_frames = data.shape[3]
        for i in range(number_frames // max_frames + 1):
            i_start = i * max_frames
            i_end = min((i + 1) * max_frames, number_frames)
            # print(f"{i}, {i_start:04d},  {i_end:04d}")
            i_data = data[:, :, :, i_start:i_end]
            samples.append(fd.Sample(i_data, reference))
    return samples, sample_raw.header


def save_data(
    data: List[np.ndarray],
    folder: str,
    file_name: str,
    header: Union[None, dict] = None,
    dtype=np.uint16,
) -> None:
    np.savez_compressed(
        os.path.join(folder, file_name),
        intensity=data[0].astype(dtype),
        lifetime_phase=data[1].astype(dtype),
        value=data[2].astype(dtype),
    )
    if header != None:
        with open(os.path.join(folder, file_name + ".json"), "w") as outfile:
            json.dump(header, outfile)


def process_samples(
    samples: List[fd.Sample],
    calibration: np.ndarray,
    factor: float = 1e3,
    dtype=np.uint16,
):
    intensity = np.array([])
    lifetime_phase = np.array([])
    value = np.array([])
    axes = [2, 1, 0]
    for i, sample in enumerate(samples):
        if i == 0:
            if sample.dc.ndim == 3:
                axes = [2, 1, 0]
            elif sample.dc.ndim == 2:
                axes = [1, 0]
            else:
                raise (ValueError("axes don't match array"))
            intensity = np.transpose(sample.dc.astype(dtype), axes=axes)
            lifetime_phase = np.transpose(
                (factor * 1e9 * sample.getlifetimephase()).astype(dtype), axes=axes
            )
            value = (factor * tau2value(lifetime_phase / factor, calibration)).astype(
                dtype
            )
        else:
            i_intensity = np.transpose(sample.dc.astype(dtype), axes=axes)
            intensity = np.append(intensity, i_intensity, axis=0)
            i_lifetime_phase = np.transpose(
                (factor * 1e9 * sample.getlifetimephase()).astype(dtype), axes=axes
            )
            lifetime_phase = np.append(lifetime_phase, i_lifetime_phase, axis=0)
            i_value = (
                factor * tau2value(i_lifetime_phase / factor, calibration)
            ).astype(dtype)
            value = np.append(value, i_value, axis=0)
        print(f"     | sub-part {i:03d}: {value.shape}")
    return intensity, lifetime_phase, value


def process_header(header: dict, **kwargs) -> dict:
    header["FLIMIMAGE"]["TIMESTAMPS in ms"] = {}
    first_offset = int(header["FLIMIMAGE"]["TIMESTAMPS"]["t0"][0])
    offset = timestamp2ms(
        header["FLIMIMAGE"]["TIMESTAMPS"]["t0"], first_offset=first_offset
    )
    for k, v in header["FLIMIMAGE"]["TIMESTAMPS"].items():
        header["FLIMIMAGE"]["TIMESTAMPS in ms"][k] = timestamp2ms(
            v, first_offset=first_offset, offset=offset
        )
    for k, v in kwargs.items():
        header[k] = v
    return header


def read_calibration(path: str) -> np.ndarray:
    return np.loadtxt(path, delimiter=";")  # Loads CSV file with calibration curve


def calibration_curve(ph, params):
    return (params["tau2"] - params["tau1"]) * np.power(10, params["pka1"] - ph) / (
        1 + np.power(10, params["pka1"] - ph)
    ) + params["tau1"] * np.power(10, params["pka2"] - ph) / (
        1 + np.power(10, params["pka2"] - ph)
    )


def produce_calibration(tau1, tau2, pka1, pka2, N, pH_range: Tuple[float, float]):
    pH_eval = np.linspace(pH_range[0], pH_range[1], N)
    params = {"tau1": tau1, "tau2": tau2, "pka1": pka1, "pka2": pka2}
    tau_cal = calibration_curve(pH_eval, params)
    return np.stack([tau_cal, pH_eval], axis=1)


def get_calibration(method: str, **kwargs):
    if method == "file":
        return read_calibration(kwargs["calibration_path"])

    elif method == "function":
        return produce_calibration(
            kwargs["tau1"],
            kwargs["tau2"],
            kwargs["pka1"],
            kwargs["pka2"],
            kwargs["N"],
            kwargs["pH_range"],
        )
    else:
        raise ValueError("Not able to retrieve calibration!")


def tau2value(x, calibration: np.ndarray):
    cal_tau = calibration[:, 0]
    cal_value = calibration[:, 1]
    return np.interp(x, cal_tau[::-1], cal_value[::-1])


def timestamp2ms(timestamp: str, first_offset: int, offset: int = 0) -> int:
    first = int(timestamp.split(" ")[0]) - first_offset
    second = int(int(timestamp.split(" ")[1][:-4]))
    combined = int(first * 4.3e9 * 1e-4 + second) - offset
    # print(first, second, combined, combined  * 1e-3)
    return combined


def process_folder(
    sample_folder: str,
    reference_path: str,
    reference_lifetime: float,
    reference_frequency: float,
    calibration_method: str,
    calibration_kwargs: dict,
    out_folder: str = "",
    filter_in: List[str] = ["fli"],
    filter_out: List[str] = [],
    max_files: int = 100,
    max_frames: int = 100,
):
    if out_folder == "":
        out_folder = sample_folder
    counter = 0
    calibration = get_calibration(calibration_method, **calibration_kwargs)
    for i, f in enumerate(os.listdir(sample_folder)):
        if counter == max_files:
            break
        if not all(crit in f for crit in filter_in):
            continue
        if any(crit in f for crit in filter_out):
            continue
        print(f" {counter:03d} > {f}")
        try:
            samples, header = get_data(
                os.path.join(sample_folder, f),
                reference_path,
                reference_lifetime,
                reference_frequency,
                max_frames=max_frames,
            )
            if samples[0].dc.ndim <= 2:
                frames = 1
            else:
                frames = samples[0].dc.shape[-1]
            print(f"     | {len(samples)} sub-parts, max. {frames} frames")
            intensity, tau, value = process_samples(samples, calibration=calibration)
            print(f"     | data with {intensity.shape} processed.")
            header = process_header(
                header,
                calibration_file=os.path.split(calibration_kwargs["calibration_path"])[
                    -1
                ],
            )
            print(f"     | header processed.")
            save_data(
                data=[intensity, tau, value],
                folder=out_folder,
                file_name=f.replace(".fli", ""),
                header=header,
            )
            print(f"     | saved to {out_folder}.\n")
        except ValueError as error:
            print(f"     | CORRUPTED - {error}\n")
        counter += 1
