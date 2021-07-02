"""
This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to <https://unlicense.org>
"""

import datetime
import math
import os
import sys
import threading
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph as pg
from PyQt5 import uic, QtWidgets, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidgetItem


class KMeans:
    """
    K-means clustering
    code from: https://dev.to/rishitdagli/build-k-means-from-scratch-in-python-2140
    """

    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for featureset in data:
                distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid - original_centroid) / original_centroid * 100.0) > self.tol:
                    print(np.sum((current_centroid - original_centroid) / original_centroid * 100.0))
                    optimized = False

            if optimized:
                break

    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification


class Window(QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        # Load GUI file
        uic.loadUi('LR1.ui', self)

        # System variables
        self.model = KMeans()
        self.dump_file = None
        self.reader_running = False
        self.dump_paused = False
        self.points = []

        # Connect GUI controls
        self.btn_load_data.clicked.connect(self.load_data)
        self.btn_stop_reading.clicked.connect(self.stop_reading)
        self.btn_pause.clicked.connect(self.pause)
        self.plot_timer = QtCore.QTimer()
        self.plot_timer.timeout.connect(self.update_plot)
        self.plot_timer.start(100)

        # Initialize table
        self.init_tables()

        # Initialize pyQtGraph charts
        self.init_charts()

        # Show GUI
        self.show()

    def init_tables(self):
        """
        Initializes table of packets and setup table (whitelist table)
        :return:
        """
        self.points_table.setColumnCount(3)
        self.points_table.verticalHeader().setVisible(False)
        self.points_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.points_table.setHorizontalHeaderItem(0, QtWidgets.QTableWidgetItem('Packet'))
        self.points_table.setHorizontalHeaderItem(1, QtWidgets.QTableWidgetItem('Time'))
        self.points_table.setHorizontalHeaderItem(2, QtWidgets.QTableWidgetItem('Data'))
        header = self.points_table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)

    def init_charts(self):
        """
        Initializes charts
        :return:
        """
        self.graphWidget.setBackground((255, 255, 255))
        self.graphWidget.showGrid(x=True, y=True, alpha=1.0)

    def update_plot(self):
        """
        Draws points over pyQTGraph
        :return:
        """
        if len(self.points) > 0 and not self.dump_paused:
            self.graphWidget.clear()

            # Find K-means clusters
            self.model.fit(np.array(self.points), k=self.slider_clusters.value())

            # Draw centroids
            centroids_x = []
            centroids_y = []
            for centroid in self.model.centroids:
                centroids_x.append(self.model.centroids[centroid][0])
                centroids_y.append(self.model.centroids[centroid][1])

            # Draw points by clusters
            color_data = np.array(range(len(self.model.classifications) + 1))
            color_map = plt.get_cmap('hsv')
            min_z = np.min(color_data)
            max_z = np.max(color_data)
            rgba_img = color_map(1.0 - (color_data - min_z) / (max_z - min_z)) * 255
            for classification in self.model.classifications:
                features_x = []
                features_y = []
                for features_et in self.model.classifications[classification]:
                    features_x.append(features_et[0])
                    features_y.append(features_et[1])
                self.graphWidget.plot(features_x, features_y, pen=None,
                                      symbolBrush=(rgba_img[classification][0],
                                                   rgba_img[classification][1],
                                                   rgba_img[classification][2]), symbolSize=5)

                max_x = np.max(features_x)
                min_x = np.min(features_x)
                max_y = np.max(features_y)
                min_y = np.min(features_y)

                self.graphWidget.plot([min_x, min_x, max_x, max_x, min_x], [min_y, max_y, max_y, min_y, min_y],
                                      pen=pg.mkPen(((
                                          rgba_img[classification][0],
                                          rgba_img[classification][1],
                                          rgba_img[classification][2]))),
                                      symbolBrush=None, symbolSize=0)

            # Plot centroids
            self.graphWidget.plot(centroids_x, centroids_y, pen=None,
                                  symbolBrush=(0, 0, 0), symbolSize=10)

            # Found lines and draw it
            points_x = np.array([item[0] for item in self.points])
            points_y = np.array([item[1] for item in self.points])
            min_x = np.min(points_x)
            min_y = np.min(points_y)
            points_x -= min_x
            points_y -= min_y
            points_x = points_x / 10
            points_y = points_y / 10
            points_image = np.zeros((int(np.max(points_y) + 1), int(np.max(points_x) + 1)), np.uint8)
            for i in range(len(points_x)):
                points_image[int(points_y[i]), int(points_x[i])] = 255
            kernel = np.ones((5, 5), np.uint8)
            points_image = cv2.dilate(points_image, kernel, iterations=2)
            min_line_length = 550
            max_line_gap = 70
            lines = cv2.HoughLinesP(points_image, 1, np.pi / 180, 100, min_line_length, max_line_gap)
            for line in lines:
                for x1, y1, x2, y2 in line:
                    self.graphWidget.plot([x1 * 10 + min_x, x2 * 10 + min_x], [y1 * 10 + min_y, y2 * 10 + min_y],
                                          pen=pg.mkPen((0, 255, 0)),
                                          symbolBrush=None, symbolSize=0)

    def load_data(self):
        """
        Loads dump file
        :return:
        """
        if not self.reader_running:
            if os.path.exists(self.data_file.text()):
                print('Loading data...')
                self.dump_file = open(self.data_file.text(), 'r')
                self.reader_running = True
                thread = threading.Thread(target=self.dump_reader)
                thread.start()
            else:
                print('File', self.data_file.text(), 'doesn\'t exist!')

    def pause(self):
        """
        Pauses data stream
        :return:
        """
        self.dump_paused = not self.dump_paused
        if self.dump_paused:
            self.btn_pause.setText('Resume')
        else:
            self.btn_pause.setText('Pause')

    def stop_reading(self):
        """
        Stops reading data from dump file
        :return:
        """
        self.reader_running = False
        self.dump_file.close()

    def dump_reader(self):
        """
        Reads dump from file
        :return:
        """
        # Clear table and data arrays
        self.points_table.setRowCount(0)

        # Create variables
        packets_read = 0
        last_packet_datetime = None

        # Continue reading
        while self.reader_running:
            # If on pause
            while self.dump_paused:
                time.sleep(0.1)

            # Read line from file
            line = self.dump_file.readline()

            # Check for line
            if line is None or len(line) < 1:
                break

            data_packet = line.split(' ')

            # Sleep defined time
            time_string = str(data_packet[0]).replace('>', '')
            if last_packet_datetime is None:
                last_packet_datetime = datetime.datetime.strptime(time_string, '%H:%M:%S.%f')
            packet_datetime = datetime.datetime.strptime(time_string, '%H:%M:%S.%f')
            time.sleep((packet_datetime - last_packet_datetime).total_seconds())
            last_packet_datetime = packet_datetime

            # Add packet to the table
            position = self.points_table.rowCount()
            self.points_table.insertRow(position)
            self.points_table.setItem(position, 0, QTableWidgetItem(str(position)))
            self.points_table.setItem(position, 1, QTableWidgetItem(str(time_string)))

            # Remove timestamp and ending from packet and convert to int
            data_packet = list(map(int, data_packet[1:][:-1]))

            self.points_table.setItem(position, 2, QTableWidgetItem(str(data_packet[0]) +
                                                                    ' ... ' + str(data_packet[-1])))

            points = []

            for i in range(len(data_packet)):
                angle = (60.0 + i * 0.36) * math.pi / 180

                x = data_packet[i] * math.sin(angle)
                y = data_packet[i] * math.cos(angle)
                points.append([x, y])

            self.points = points.copy()

            # Increment counter
            packets_read += 1

        self.dump_file.close()
        print('File reading stopped. Read', packets_read, 'packets')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('fusion')
    win = Window()
    sys.exit(app.exec_())
