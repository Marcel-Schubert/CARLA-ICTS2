import matplotlib.pyplot as plt
import numpy as np
import matplotlib.collections as mcoll
import matplotlib.path as mpath

# #CONSTANTS
# NPOINTS = 2
# COLOR='blue'
# RESFACT=10
# MAP='winter'
# cm = plt.get_cmap(MAP)
# fig, ax = plt.subplots()
# ax.set_prop_cycle('color', [cm(1.*i/(NPOINTS-1)) for i in range(NPOINTS-1)])



interactive_scenarios = [
    "01_int",
    "02_int",
    "03_int",
    "04_int",
    "05_int",
    "06_int"
]

non_interactive_scenarios = [
    "01_non_int",
    "02_non_int",
    "03_non_int",
    "04_non_int",
    "05_non_int",
    "06_non_int"
    ]

all_int = np.empty(0)
all_non_int = np.empty(0)
all_int_car = np.empty(0)
all_non_int_car = np.empty(0)

for scenario in interactive_scenarios:
    with open(f"./P3VI/data/{scenario}.npy", "rb") as f:
        arr = np.load(f, allow_pickle=True)

        # Remove empties:
        to_delete = np.unique(np.where(arr[:, :, 0] == 0)[0])

        arr = np.delete(arr, to_delete, 0)
        print("PED", arr.shape)
        enum_conv = lambda t: t.value
        vfunc = np.vectorize(enum_conv)
        arr[:, :, 2] = vfunc(arr[:, :, 2])
        arr[:, :, 3] = vfunc(arr[:, :, 3])
        arr = arr.astype(np.float32)
        np.save(f"./P3VI/data/{scenario}_cleaned.npy", arr, allow_pickle=False)


        plt.plot(arr[:, :, 0].T, arr[:, :, 1].T, label=scenario)
        plt.title(scenario)
        # plt.show()


        if all_int.size == 0:
            all_int = arr
        else:
            all_int = np.concatenate((all_int, arr))

    with open(f"./P3VI/data/{scenario}_car.npy", "rb") as f:
        arr = np.load(f, allow_pickle=True)

        arr = np.delete(arr, to_delete, 0)
        print("CAR", arr.shape)

        # plt.plot(arr[:, :, 0].T, arr[:, :, 1].T, label=scenario)
        # plt.title(scenario+" car")
        plt.show()
        np.save(f"./P3VI/data/{scenario}_cleaned_car.npy", arr, allow_pickle=False)

        if all_int_car.size == 0:
            all_int_car = arr
        else:
            all_int_car = np.concatenate((all_int_car, arr))
plt.show()

for scenario in non_interactive_scenarios:
    with open(f"./P3VI/data/{scenario}.npy", "rb") as f:
        arr = np.load(f, allow_pickle=True)


        # Remove empties:
        to_delete = np.unique(np.where(arr[:, :, 0] == 0)[0])

        arr = np.delete(arr, to_delete, 0)
        print("PED", arr.shape)

        enum_conv = lambda t: t.value
        vfunc = np.vectorize(enum_conv)
        arr[:, :, 2] = vfunc(arr[:, :, 2])
        arr[:, :, 3] = vfunc(arr[:, :, 3])
        arr = arr.astype(np.float32)

        np.save(f"./P3VI/data/{scenario}_cleaned.npy", arr, allow_pickle=False)

        plt.plot(arr[:, :, 0].T, arr[:, :, 1].T, label=scenario)
        plt.title(scenario)
        # plt.show()

        if all_non_int.size == 0:
            all_non_int = arr
        else:
            all_non_int = np.concatenate((all_non_int, arr))

    with open(f"./P3VI/data/{scenario}_car.npy", "rb") as f:
        arr = np.load(f, allow_pickle=True)
        arr = np.delete(arr, to_delete, 0)
        print("CAR", arr.shape)

        np.save(f"./P3VI/data/{scenario}_cleaned_car.npy", arr, allow_pickle=False)
        if all_non_int_car.size == 0:
            all_non_int_car = arr
        else:
            all_non_int_car = np.concatenate((all_non_int_car, arr))
plt.show()

print(all_int.shape)
print(all_non_int.shape)
print(all_int_car.shape)
print(all_non_int_car.shape)

# plt.plot(all_int[:, :, 0].T, all_int[:, :, 1].T)
# plt.title("All interactive")
# plt.show()
#
# plt.plot(all_non_int[:, :, 0].T, all_non_int[:, :, 1].T)
# plt.title("All non interactive")
# plt.show()



np.save(f"./P3VI/data/all_int.npy", all_int, allow_pickle=True)
np.save(f"./P3VI/data/all_non_int.npy", all_non_int, allow_pickle=True)
np.save(f"./P3VI/data/all_int_car.npy", all_int_car, allow_pickle=True)
np.save(f"./P3VI/data/all_non_int_car.npy", all_non_int_car, allow_pickle=True)
