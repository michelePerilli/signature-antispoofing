import os
import pickle
import numpy as np
from pandas import DataFrame
from PIL import Image
from scipy.fft import fft
from skimage.draw import draw
import time
import pywt
from sklearn.decomposition import PCA

SLICES = 16  # Circular Grid Sectors
USERS = 300
RADIUS = 150  # Circular Grid Radius
SIGNATURES = 6  # Signatures per user; set to 6 if you use 'dataset_test'


def create_folder(path):
    if not (os.path.exists(path)):  # If not exists
        os.mkdir(path)


class CircularGridFeatureExtraction:

    def __init__(self, dataset, pickle_folder='.pickles\\', csv_folder='.csv\\'):
        self.pickle_folder = pickle_folder
        self.csv_folder = csv_folder
        self.dataset = dataset  # Signatures dataset folder
        self.slice_center = []
        new_im = Image.new("1", (RADIUS * 2, RADIUS * 2), 'white')
        mask = self.create_slice_mask(new_im, (RADIUS, RADIUS), RADIUS)
        new_im = np.asmatrix(new_im).copy()
        new_im[~mask] = 0
        self.total_pixels_inside_sector = new_im[new_im == 1].size
        self.alpha_max = (2 * np.math.pi) / SLICES


    # Extraction of the features from signatures
    def extract(self):
        startTime = time.time()
        for user in os.listdir(self.dataset):
            userTime = time.time()
            user_folder_path = self.dataset + user + '\\'

            XPD = []  # Vector of all images density pixel
            XDGC = []  # Vector of all images gravity center distance
            XAGC = []  # Vector of all images gravity center angle
            horizontals, verticals = [], []  # Wavelet outputs
            for file in os.listdir(user_folder_path):
                signatureTime = time.time()
                file_path = user_folder_path + file
                image = self.prepare_image(file_path)

                # Wavelet transform
                LL, (LH, HL, HH) = pywt.dwt2(image, 'db4')

                # Hotelling transform (dimensionality reduction)
                pca = PCA(n_components=1)
                pca.fit(LH)
                horizontals.append(pca.components_[0])
                pca = PCA(n_components=1)
                pca.fit(HL)
                verticals.append(pca.components_[0])

                XPDi, XDGCi, XAGCi = [], [], []

                for i in range(SLICES):
                    mask = self.create_slice_mask(image, (RADIUS, RADIUS), RADIUS, i=i)
                    masked_image = np.asmatrix(image).copy()
                    masked_image[~mask] = True
                    self.slice_center = self.calculate_center_of_mass(masked_image)

                    XPDi.append(self.get_pixel_density_distribution(masked_image))
                    XDGCi.append(self.get_gravity_center_distance())
                    XAGCi.append(self.get_gravity_center_angle(i))

                XPD.append(XPDi)
                XDGC.append(XDGCi)
                XAGC.append(XAGCi)
                print("\t\t%s extracted: %.2f seconds" % (file, time.time() - signatureTime))
            print("\tUser %s processed: %.2f seconds" % (user, time.time() - userTime))
            self.save_pickle_user(user, XPD, XDGC, XAGC, horizontals, verticals)
        print("\nExtraction process ended: %.2f seconds" % (time.time() - startTime))

    # Saving temporary files for backup
    def save_pickle_user(self, user, XPD, XDGC, XAGC, horizontals, verticals):
        user_folder = self.pickle_folder + user + '\\'
        create_folder(user_folder)
        pickle.dump(XPD, open(user_folder + 'XPD.bin', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(XDGC, open(user_folder + 'XDGC.bin', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(XAGC, open(user_folder + 'XAGC.bin', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(horizontals, open(user_folder + 'LH.bin', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(verticals, open(user_folder + 'HL.bin', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    # Loading temporary files for backup
    def load_user_data(self, user):
        user_folder = self.pickle_folder + user + '\\'
        XPD = pickle.load(open(user_folder + 'XPD.bin', 'rb'))
        XDGC = pickle.load(open(user_folder + 'XDGC.bin', 'rb'))
        XAGC = pickle.load(open(user_folder + 'XAGC.bin', 'rb'))
        horizontals = pickle.load(open(user_folder + 'LH.bin', 'rb'))
        verticals = pickle.load(open(user_folder + 'HL.bin', 'rb'))
        return XPD, XDGC, XAGC, horizontals, verticals

    # Generate CSVs file containing the extracted features
    def generate_csv(self):
        for user in os.listdir(self.pickle_folder):
            feature = []
            user_folder = self.csv_folder + user + '\\'
            XPD, XDGC, XAGC, horizontals, verticals = self.load_user_data(user)
            create_folder(user_folder)
            for sign in range(SIGNATURES):
                record = np.concatenate((
                    [abs(element) for element in (fft(XPD[sign][::]).real[:SLICES // 2 + 1])],
                    [abs(element) for element in (fft(XDGC[sign][::]).real[:SLICES // 2 + 1])],
                    [abs(element) for element in (fft(XAGC[sign][::]).real[:SLICES // 2 + 1])],
                    horizontals[sign][::],
                    verticals[sign][::]
                ))

                feature.append(record)
            DataFrame(feature, range(SIGNATURES)).to_csv(user_folder + 'features.csv')

    ##
    #
    # Static graphometric features considered:
    #
    # XPD: Pixel Density Distribution
    # XDGC: Gravity Center Distance
    # XAGC: Gravity Center Angle
    #

    # Local feature 1: black pixels inside the sector / total pixels inside the sector
    def get_pixel_density_distribution(self, sector):
        black_pixels = sector[sector == 0].size
        return black_pixels / self.total_pixels_inside_sector

    # Local feature 2: distance between centers / grid radius
    def get_gravity_center_distance(self):
        if self.slice_center == (0, 0):
            return 0
        else:
            cathetus_1 = RADIUS - self.slice_center[0]
            cathetus_2 = RADIUS - self.slice_center[1]

            # Pythagoras
            dGCi = np.sqrt(cathetus_1 ** 2 + cathetus_2 ** 2)
            if np.isnan(dGCi):
                dGCi = 0
            return dGCi / RADIUS

    # Local feature 3: angle between slice center and slice / angle of slice
    def get_gravity_center_angle(self, i):
        if self.slice_center == (0, 0):
            return 0
        else:
            angle = np.arccos(abs(RADIUS - self.slice_center[0]) / (self.get_gravity_center_distance() * RADIUS))
            if np.isnan(angle):
                return 0
            xgca = 0
            if i in range(0, 4):
                xgca = angle - self.alpha_max * i
            elif i in range(4, 8):
                xgca = self.alpha_max - (angle - (SLICES // 2 - 1 - i) * self.alpha_max)
            elif i in range(8, 12):
                xgca = angle - self.alpha_max * (i - SLICES // 2)
            elif i in range(12, 16):
                xgca = self.alpha_max - (angle - (SLICES - 1 - i) * self.alpha_max)
            return xgca / self.alpha_max

    ##
    # Utils functions
    #

    def prepare_image(self, file_path):
        image = Image.open(file_path).convert('1')

        ratio = float(RADIUS * 2 - RADIUS // 4) / max(image.size)
        new_size = tuple([int(x * ratio) for x in image.size])
        # new_size = tuple([RADIUS + RADIUS // 2, RADIUS + RADIUS // 2])

        image = image.resize(new_size, Image.ANTIALIAS)
        center = self.calculate_center_of_mass(image)

        offset = int(abs(center[0] - RADIUS)), int(abs(center[1] - RADIUS))

        new_im = Image.new("1", (RADIUS * 2, RADIUS * 2), 'white')
        new_im.paste(image, offset)
        # new_im.filter(ImageFilter.M)
        return new_im

    def calculate_center_of_mass(self, image):
        image = np.asmatrix(image)
        x, y = [], []

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if not image[i, j]:
                    x.append(j)
                    y.append(i)
        if len(x) > 0 and len(y) > 0:
            return [np.sum(x) / len(x), np.sum(y) / len(y)]
            # return [np.average(x), np.average(y)]
        else:
            return 0, 0

    def create_slice_mask(self, img, c, radius=RADIUS, i=0, N=SLICES):
        # Random image
        image = np.asmatrix(img)

        # --- coordinate specification

        r0, c0 = c[1], c[0]  # circle center (row, column)
        R = radius  # circle radius

        theta0 = (2 * i * np.math.pi) / N  # angle #1 for arc
        theta1 = (2 * (i + 1) * np.math.pi) / N  # angle #2 for arc

        # Above, I provide two angles, but you can also just give the two
        # coordinates below directly

        r1, c1 = r0 - 1.5 * R * np.sin(theta0), c0 + 1.5 * R * np.cos(theta0)  # arc coord #1
        r2, c2 = r0 - 1.5 * R * np.sin(theta1), c0 + 1.5 * R * np.cos(theta1)  # arc coord #2

        # --- mask calculation

        mask_circle = np.zeros(image.shape[:2], dtype=bool)
        mask_poly = np.zeros(image.shape[:2], dtype=bool)

        rr, cc = draw.disk(c, R, shape=mask_circle.shape)
        # rr, cc = draw.circle(c[0], c[1], R, shape=mask_circle.shape)  # Use this if "draw.disk" doesn't work
        mask_circle[rr, cc] = 1

        rr, cc = draw.polygon([r0, r1, r2, r0],
                              [c0, c1, c2, c0], shape=mask_poly.shape)
        mask_poly[rr, cc] = 1

        mask = mask_circle & mask_poly

        return mask


def compute_feature_extraction(dataset, path_pickle, path_csv):
    create_folder(pickle_path)
    create_folder(csv_path)
    extractor = CircularGridFeatureExtraction(dataset, path_pickle, path_csv)
    # extractor.extract()
    extractor.generate_csv()


def print_elapsed_time(seconds):
    ore = seconds // 3600
    minuti = (seconds % 3600) // 60
    secondi = seconds % 60
    print('\nCompletato in: %.0d ore, %.0d minuti, %.0d secondi' % (ore, minuti, secondi))


dataset_path = 'dataset\\'
pickle_path = '.pickles\\'
csv_path = '.csv\\'
dataset_path_test = 'dataset_test\\'
pickle_path_test = '.pickles_test\\'
csv_path_test = '.csv_test\\'

if __name__ == '__main__':
    start = time.time()

    # compute_feature_extraction(dataset_path, pickle_path, csv_path)
    compute_feature_extraction(dataset_path_test, pickle_path_test, csv_path_test)

    print_elapsed_time(time.time() - start)



