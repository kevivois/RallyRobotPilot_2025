

if __name__ == "__main__":
    import numpy
    import pickle
    import lzma


    with lzma.open("record_0.npz", "rb") as file:
        data = pickle.load(file)

        print("Read", len(data), "snapshots")
        print(data[0].image)
        print([len(e.raycast_distances) for e in data])
