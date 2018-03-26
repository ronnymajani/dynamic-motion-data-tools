class DataSetContract:
    VERSION = "1.0"

    JSON_DATASET_TEMPLATE = "../Templates/data_v%s_template_dataset.json" % VERSION
    JSON_DIGITSET_TEMPLATE = "../Templates/data_v%s_template_digitset.json" % VERSION

    METADATA = "dataset_metadata"
    DIGITSETS = "digitsets"

    class Metadata:
        VERSION = "version"
        MISC = "misc"

        class Device:
            NAME = "device_name"
            SAMPLING_RATE = "sampling_rate"
            RESOLUTION = "device_resolution"
            PEN_PRESSURE = "pressure_resolution"

            RESOLUTION_WIDTH = "W"
            RESOLUTION_HEIGHT = "H"

    class DigitSet:
        METADATA = "digitset_metadata"
        DIGITS = "digits"
        digits = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")

        class Frame:
            columns = ('X', 'Y', 'P', 'dt')
            # The indices of each value in a sample
            indices = {
                "X": 0,
                "Y": 1,
                "P": 2,
                "dt": 3
            }

        class Metadata:
            USER_AGE = "user_age"
            USER_SEX = "user_sex"
            USER_HAND = "user_hand"

            USER_SEX_MALE = "male"
            USER_SEX_FEMALE = "female"
            USER_HAND_RIGHT = "right"
            USER_HAND_LEFT = "left"
