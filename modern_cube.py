import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import OpenGL.GL.shaders
import numpy
import pyrr
from PIL import Image
import time 
import math

import cv2
from cv2 import aruco   

ARUCO_DICT=aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)

def loadTextures():
        
    texture = glGenTextures(1)
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D_ARRAY, texture)

    glTexStorage3D(GL_TEXTURE_2D_ARRAY, 6, GL_RGB8, 320, 320, 8) 
    
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

    
    for i in range(6):
        image = Image.new('RGB', (320,320), (255,255,255))
        image.paste(Image.fromarray(aruco.drawMarker(ARUCO_DICT,i+2,300)).convert("RGB"), (10,10))

        ix = image.size[0]
        iy = image.size[1]

        image = image.tobytes("raw", "RGB", 0, -1)

       	glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0,
                        0, 0, i,
                        ix, iy ,1,
                        GL_RGB, GL_UNSIGNED_BYTE,
                        image)


        # glPixelStorei(GL_UNPACK_ALIGNMENT,1)
        # glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA , ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, image)
        #glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
        #glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
        #glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        #glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        #glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        #glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)
    
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glGenerateMipmap(GL_TEXTURE_2D_ARRAY)

    glEnable(GL_TEXTURE_2D)

def detectTag(img, camera_matrix, dist_coeffs, ra):
    markerLength = 4   # Here, our measurement unit is centimetre.
    arucoParams = aruco.DetectorParameters_create()
    
    imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)    # aruco.detectMarkers() requires gray image
    corners, ids, rejectedImgPoints = aruco.detectMarkers(imgGray, ARUCO_DICT, parameters=arucoParams) # Detect aruco

    sstr = f"0, {ra[0]} , {ra[1]}, {ra[2]}"
      
    
    if ids is not None: # if aruco marker detected
        rvec, tvec, pts = aruco.estimatePoseSingleMarkers(corners, markerLength, camera_matrix, dist_coeffs) # posture estimation from a single marker
        imgWithAruco = aruco.drawDetectedMarkers(img, corners, ids, (0,255,0))
        for i in range(len(tvec)):
            imgWithAruco = aruco.drawAxis(imgWithAruco, camera_matrix, dist_coeffs, rvec[i], tvec[i], 4)    # axis length 100 can be changed according to your requirement

        for i in list(range(2,8))+[768]:
            idx = numpy.where(ids == [i])[0]
            
            if idx:
                n = idx[0]
                angs = numpy.array(rvec[n][0]) * 360.0 / math.pi
                sstr+=f", {i}, {angs[0]}, {angs[1]}, {angs[2]}"
            else:
                sstr+=', '*4
    
        print(sstr)

    else:   # if aruco marker is NOT detected
        imgWithAruco = img  # assign imRemapped_color to imgWithAruco directly



    cv2.imshow("aruco", imgWithAruco)   # display

def main():

    # initialize glfw
    if not glfw.init():
        return

    window = glfw.create_window(1000, 1000, "My OpenGL window", None, None)
    window.visible = False

    if not window:
        glfw.terminate()
        return
    

    glfw.make_context_current(window)

    #        positions         colors          texture coords
    cube = [-0.5, -0.5,  0.5,  1.0, 1.0, 1.0,  0.0, 0.0, 0.0,
             0.5, -0.5,  0.5,  1.0, 1.0, 1.0,  1.0, 0.0, 0.0,
             0.5,  0.5,  0.5,  1.0, 1.0, 1.0,  1.0, 1.0, 0.0,
            -0.5,  0.5,  0.5,  1.0, 1.0, 1.0,  0.0, 1.0, 0.0,

            -0.5, -0.5, -0.5,  1.0, 1.0, 1.0,  0.0, 0.0, 1.0,
             0.5, -0.5, -0.5,  1.0, 1.0, 1.0,  1.0, 0.0, 1.0,
             0.5,  0.5, -0.5,  1.0, 1.0, 1.0,  1.0, 1.0, 1.0,
            -0.5,  0.5, -0.5,  1.0, 1.0, 1.0,  0.0, 1.0, 1.0,

             0.5, -0.5, -0.5,  1.0, 1.0, 1.0,  0.0, 0.0, 2.0,
             0.5,  0.5, -0.5,  1.0, 1.0, 1.0,  1.0, 0.0, 2.0,
             0.5,  0.5,  0.5,  1.0, 1.0, 1.0,  1.0, 1.0, 2.0,
             0.5, -0.5,  0.5,  1.0, 1.0, 1.0,  0.0, 1.0, 2.0,

            -0.5,  0.5, -0.5,  1.0, 1.0, 1.0,  0.0, 0.0, 3.0,
            -0.5, -0.5, -0.5,  1.0, 1.0, 1.0,  1.0, 0.0, 3.0,
            -0.5, -0.5,  0.5,  1.0, 1.0, 1.0,  1.0, 1.0, 3.0,
            -0.5,  0.5,  0.5,  1.0, 1.0, 1.0,  0.0, 1.0, 3.0,

            -0.5, -0.5, -0.5,  1.0, 1.0, 1.0,  0.0, 0.0, 4.0,
             0.5, -0.5, -0.5,  1.0, 1.0, 1.0,  1.0, 0.0, 4.0,
             0.5, -0.5,  0.5,  1.0, 1.0, 1.0,  1.0, 1.0, 4.0,
            -0.5, -0.5,  0.5,  1.0, 1.0, 1.0,  0.0, 1.0, 4.0,

             0.5,  0.5, -0.5,  1.0, 1.0, 1.0,  0.0, 0.0, 5.0,
            -0.5,  0.5, -0.5,  1.0, 1.0, 1.0,  1.0, 0.0, 5.0,
            -0.5,  0.5,  0.5,  1.0, 1.0, 1.0,  1.0, 1.0, 5.0,
             0.5,  0.5,  0.5,  1.0, 1.0, 1.0,  0.0, 1.0, 5.0]

    cube = numpy.array(cube, dtype = numpy.float32)

    indices = [ 0,  1,  2,  2,  3,  0,
                4,  5,  6,  6,  7,  4,
                8,  9, 10, 10, 11,  8,
               12, 13, 14, 14, 15, 12,
               16, 17, 18, 18, 19, 16,
               20, 21, 22, 22, 23, 20]

    indices = numpy.array(indices, dtype= numpy.uint32)

    vertex_shader = """
    #version 330
    in layout(location = 0) vec3 position;
    in layout(location = 1) vec3 color;
    in layout(location = 2) vec2 textureCoords;
    in layout(location = 3) float textureid;
    uniform mat4 transform;
    uniform mat4 proj;
    out vec3 newColor;
    out vec2 newTexture;
    out float newtextureid;
    
    void main()
    {
        gl_Position = proj * transform * vec4(position, 1.0f);
        newColor = color;
        newTexture = textureCoords;
        newtextureid = textureid;
    }
    """

    fragment_shader = """
    #version 330
    #extension GL_EXT_texture_array : enable
    in vec3 newColor;
    in vec2 newTexture;
    in float newtextureid;

    uniform sampler2DArray samplerTexture;

    out vec3 outColor;
    
    void main()
    {
        outColor = (texture(samplerTexture, vec3(newTexture, newtextureid))  * vec4(newColor, 1.0f)).rgb;
        //outColor = texture(samplerTexture, newTexture)  * vec4(newColor, 1.0f);
    }
    """
    shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
                                              OpenGL.GL.shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER))

    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, cube.itemsize * len(cube), cube, GL_STATIC_DRAW)

    EBO = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.itemsize * len(indices), indices, GL_STATIC_DRAW)

    #position
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, cube.itemsize * 9, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    #color
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, cube.itemsize * 9, ctypes.c_void_p(12))
    glEnableVertexAttribArray(1)
    #texture
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, cube.itemsize * 9, ctypes.c_void_p(24))
    glEnableVertexAttribArray(2)
    #textureid
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, cube.itemsize * 9, ctypes.c_void_p(32))
    glEnableVertexAttribArray(3)


    loadTextures()
    """
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    # Set the texture wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    # Set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    # load image
    image = Image.open("res/crate.jpg")
    img_data = numpy.array(list(image.getdata()), numpy.uint8)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
    """



    glUseProgram(shader)

    glClearColor(0.2, 0.2, 0.2, 1.0)

    glShadeModel (GL_SMOOTH)
    glEnable(GL_DEPTH_TEST)

    LightAmbient = [ 0.5, 0.5, 0.5, 1.0 ]
    LightDiffuse = [ 0.5, 0.5, 0.5, 1.0 ]
    LightPosition = [ 5.0, 25.0, 15.0, 1.0 ]

    glLightfv(GL_LIGHT1 ,GL_AMBIENT, LightAmbient)
    glLightfv(GL_LIGHT1 ,GL_DIFFUSE, LightDiffuse)
    glLightfv(GL_LIGHT1 ,GL_POSITION, LightPosition)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT1)

    #glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

    (xa,ya,za) = (0,0,0)

    while not glfw.window_should_close(window):
        
        (w,h) =glfw.get_window_size(window)

        glfw.poll_events()
        #time.sleep(0.2)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        #(xa,ya,za) = (12,45,77)
        #(xa,ya,za) = numpy.random.uniform(size=3)*360.0
        (xa,ya,za) = numpy.array([0.8,0.3,0.2])*glfw.get_time()

        rot_x = pyrr.Matrix44.from_x_rotation(xa )
        rot_y = pyrr.Matrix44.from_y_rotation(ya)
        rot_z = pyrr.Matrix44.from_z_rotation(za )

        (dx,dy,dz) = (0.0,0.0,-0.2)
        #(dx,dy,dz) =  numpy.random.uniform(-0.5,0.5,size=3)

        trans = pyrr.matrix44.create_from_translation([dx, dy, -2+dz])

        proj = pyrr.matrix44.create_perspective_projection_matrix(80.0,1.0,0.1,1000.0)
  
        rot = numpy.array(rot_x * rot_y * rot_z * trans)

        ra = numpy.array([xa,ya,za]) * 360.0 / math.pi % 360.0

        transformLoc = glGetUniformLocation(shader, "transform")
        projLoc = glGetUniformLocation(shader, "proj")

        glUniformMatrix4fv(transformLoc, 1, GL_FALSE, rot)
        glUniformMatrix4fv(projLoc, 1, GL_FALSE, proj)

        glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

        glfw.swap_buffers(window)

        pixels = glReadPixelsui(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE)
        im = numpy.frombuffer(pixels,dtype='uint8').reshape(h,w,3)
        im = cv2.flip(im,0)

        pm = numpy.array([[w,0,w/2],
                          [0,h,h/2],
                          [0,0,1]])

        dist = numpy.array([0.0,0.0,0.0,0.0]).reshape(4,1)

        detectTag(im, pm, dist,ra)


    glfw.terminate()

if __name__ == "__main__":
    main()