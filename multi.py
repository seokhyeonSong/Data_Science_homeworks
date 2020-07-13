import cv2
import numpy as np

def find_pattern_aruco(image, aruco_marker, sigs):
    # converting image to black and white
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # adaptive thresholding for robustness against varying lighting
    thresholded = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 10)
    h, w = aruco_marker.shape

    _, contours, _ = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        if approx.shape[0] == 4:
            x1 = approx[0][0][0]
            x2 = approx[1][0][0]
            y1 = approx[0][0][1]
            y2 = approx[1][0][1]

            norm = (x1 - x2) ** 2 + (y1 - y2) ** 2
            # constraint on minimum edge size of quad
            if norm > 100:
                temp_sig = get_bit_sig(gray, approx)
                match1 = match_sig(sigs[0], temp_sig)
                match2 = match_sig(sigs[1], temp_sig)
                match3 = match_sig(sigs[2], temp_sig)
                match4 = match_sig(sigs[3], temp_sig)

                if (match1 or match2 or match3 or match4):
                    dst_pts = approx
                    if match1:
                        src_pts = np.array([[0, 0], [0, w], [h, w], [h, 0]])
                    if match2:
                        src_pts = np.array([[0, w], [h, w], [h, 0], [0, 0]])
                    if match3:
                        src_pts = np.array([[h, w], [h, 0], [0, 0], [0, w]])
                    if match4:
                        src_pts = np.array([[h, 0], [0, 0], [0, w], [h, w]])

                    cv2.drawContours(image, [approx], 0, (0, 0, 255), 2)  # mark red outline for found marker

                    return src_pts, dst_pts, True

    # reaching here implies nothing was found
    return None, None, False
def get_bit_sig(image, contour_pts, thresh = 127):
    ans = []

    #getting all the 4 corners of the quad
    a, b = contour_pts[0][0]
    c, d = contour_pts[1][0]
    e, f = contour_pts[3][0]
    g, h = contour_pts[2][0]

    for i in range(8):
        for j in range(8):
            #using bilinear interpolation to find the coordinate using fractional contributions of the corner 4 points
            f1 = float(i)/8 + 1./16 #fraction1
            f2 = float(j)/8 + 1./16 #fraction2

            #finding the intermediate coordinates
            upper_x = (1-f1)*a + f1*(c)
            lower_x = (1-f1)*e + f1*(g)
            upper_y = (1-f1)*b + f1*d
            lower_y = (1-f1)*(f) + f1*(h)

            x = int( (1-f2)*upper_x + (f2)*lower_x )
            y = int( (1-f2)*upper_y + (f2)*lower_y )

            #thresholding
            if image[y][x] >= 127:
                ans.append(1)
            else:
                ans.append(0)
    return ans


def get_extended_RT(A, H):
    # finds r3 and appends
    # A is the intrinsic mat, and H is the homography estimated
    H = np.float64(H)  # for better precision
    A = np.float64(A)
    R_12_T = np.linalg.inv(A).dot(H)

    r1 = np.float64(R_12_T[:, 0])  # col1
    r2 = np.float64(R_12_T[:, 1])  # col2
    T = R_12_T[:, 2]  # translation

    # ideally |r1| and |r2| should be same
    # since there is always some error we take square_root(|r1||r2|) as the normalization factor
    norm = np.float64(math.sqrt(np.float64(np.linalg.norm(r1)) * np.float64(np.linalg.norm(r2))))

    r3 = np.cross(r1, r2) / (norm)
    R_T = np.zeros((3, 4))
    R_T[:, 0] = r1
    R_T[:, 1] = r2
    R_T[:, 2] = r3
    R_T[:, 3] = T
    return R_T


def augment(img, obj, projection, template, color=False, scale=4):
    # takes the captureed image, object to augment, and transformation matrix
    # adjust scale to make the object smaller or bigger, 4 works for the fox

    h, w = template.shape
    vertices = obj.vertices
    img = np.ascontiguousarray(img, dtype=np.uint8)

    # blacking out the aruco marker
    a = np.array([[0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0]], np.float64)
    imgpts = np.int32(cv2.perspectiveTransform(a.reshape(-1, 1, 3), projection))
    cv2.fillConvexPoly(img, imgpts, (0, 0, 0))

    # projecting the faces to pixel coords and then drawing
    for face in obj.faces:
        # a face is a list [face_vertices, face_tex_coords, face_col]
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])  # -1 because of the shifted numbering
        points = scale * points
        points = np.array([[p[2] + w / 2, p[0] + h / 2, p[1]] for p in points])  # shifted to centre
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)  # transforming to pixel coords
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, (50, 50, 50))
        else:
            cv2.fillConvexPoly(img, imgpts, face[-1])

    return img


A = [[1019.37187, 0, 618.709848], [0, 1024.2138, 327.280578], [0, 0, 1]]  # hardcoded intrinsic matrix for my webcam
A = np.array(A)


def main():
    marker_colored = cv2.imread('data/m1_flip_new.png')
    marker_colored = cv2.resize(marker_colored, (480, 480), interpolation=cv2.INTER_CUBIC)
    marker = cv2.cvtColor(marker_colored, cv2.COLOR_BGR2GRAY)

    cv2.namedWindow("webcam")
    vc = cv2.VideoCapture(0)

    h, w = marker.shape
    # considering all 4 rotations
    marker_sig1 = aruco.get_bit_sig(marker, np.array([[0, 0], [0, w], [h, w], [h, 0]]).reshape(4, 1, 2))
    marker_sig2 = aruco.get_bit_sig(marker, np.array([[0, w], [h, w], [h, 0], [0, 0]]).reshape(4, 1, 2))
    marker_sig3 = aruco.get_bit_sig(marker, np.array([[h, w], [h, 0], [0, 0], [0, w]]).reshape(4, 1, 2))
    marker_sig4 = aruco.get_bit_sig(marker, np.array([[h, 0], [0, 0], [0, w], [h, w]]).reshape(4, 1, 2))

    sigs = [marker_sig1, marker_sig2, marker_sig3, marker_sig4]

    rval, frame = vc.read()
    h2, w2, _ = frame.shape

    h_canvas = max(h, h2)
    w_canvas = w + w2

    while rval:
        rval, frame = vc.read()  # fetch frame from webcam
        key = cv2.waitKey(20)
        if key == 27:
            break

        canvas = np.zeros((h_canvas, w_canvas, 3), np.uint8)  # final display
        canvas[:h, :w, :] = marker_colored  # marker for reference

        success, H = aruco.find_homography_aruco(frame, marker, sigs)
        # success = False
        if not success:
            # print('homograpy est failed')
            canvas[:h2, w:, :] = np.flip(frame, axis=1)
            cv2.imshow("webcam", canvas)
            continue

        R_T = get_extended_RT(A, H)
        transformation = A.dot(R_T)

        augmented = np.flip(augment(frame, obj, transformation, marker, True), axis=1)  # flipped for better control
        canvas[:h2, w:, :] = augmented
        cv2.imshow("webcam", canvas)