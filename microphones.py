"""Microphone discovery and selection.

Windows lists every microphone once per audio API (MME, DirectSound, WASAPI, WDM-KS).
We show only one API's view: DirectSound (full names, resamples to 16 kHz for us),
falling back to MME, then whatever else is available.
"""

import sounddevice as sd

PREFERRED_APIS = ["Windows DirectSound", "MME"]
SKIP_NAMES = ("Sound Mapper", "Primary Sound Capture Driver")


def list_input_devices(sample_rate):
    """Return [(portaudio_index, name)] of usable microphones."""
    apis = [api["name"] for api in sd.query_hostapis()]
    api_order = [a for a in PREFERRED_APIS if a in apis] + [a for a in apis if a not in PREFERRED_APIS]

    for api_name in api_order:
        devices = []
        for idx, dev in enumerate(sd.query_devices()):
            if (dev["max_input_channels"] < 1 or apis[dev["hostapi"]] != api_name
                    or any(skip in dev["name"] for skip in SKIP_NAMES)):
                continue
            try:
                sd.check_input_settings(device=idx, samplerate=sample_rate, channels=1, dtype="float32")
            except Exception:
                continue
            devices.append((idx, dev["name"]))
        if devices:
            return devices
    return []


def _default_position(devices):
    """Position in `devices` of the system default mic (matched by name across APIs)."""
    try:
        default_name = sd.query_devices(kind="input")["name"]
    except Exception:
        return 0
    for pos, (_, name) in enumerate(devices):
        if name.startswith(default_name) or default_name.startswith(name):  # MME truncates names
            return pos
    return 0


def print_devices(devices):
    default = _default_position(devices)
    for pos, (_, name) in enumerate(devices):
        print(f"  {pos + 1}. {name}{'  (default)' if pos == default else ''}")


def choose_device(sample_rate, requested=None):
    """Pick a mic by list number or name substring; prompt interactively if `requested` is None.

    Returns (portaudio_index, name).
    """
    devices = list_input_devices(sample_rate)
    if not devices:
        raise SystemExit("No usable microphone found.")

    def resolve(choice):
        choice = choice.strip()
        if not choice:
            return devices[_default_position(devices)]
        if choice.isdigit() and 1 <= int(choice) <= len(devices):
            return devices[int(choice) - 1]
        matches = [d for d in devices if choice.lower() in d[1].lower()]
        return matches[0] if len(matches) == 1 else None

    if requested is not None:
        if (picked := resolve(requested)) is None:
            print(f"No single microphone matches '{requested}'. Available:")
            print_devices(devices)
            raise SystemExit(1)
        return picked

    if len(devices) == 1:
        return devices[0]

    print("Available microphones:")
    print_devices(devices)
    while True:
        if (picked := resolve(input("Choose microphone [Enter = default]: "))) is not None:
            return picked
        print("  invalid choice, enter a number from the list")
