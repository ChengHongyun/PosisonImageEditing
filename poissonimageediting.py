import numpy as np
import scipy.sparse as sp
import cv2 as cv
import scipy.sparse.linalg
import scipy.io
import os



def get_outline(NormalizedMask_npArr):
  """
    @input: binary mask image
    @return:  binary contuor image
  """
  Mask_01Arr = np.array(NormalizedMask_npArr, dtype=np.uint8)
  ErodedMask_01Arr = cv.erode(Mask_01Arr, np.ones((3, 3), np.uint8)) #去噪、去粘连
  Outline = Mask_01Arr * (1 - ErodedMask_01Arr) # 0~1, uint8 

  return Outline




def get_edge(gray, weight=2):

  '''
  input: gray image
  return: binary edge mask

  --> weight value can change thickness, 2 or 3 is good for example picture.

  '''


  ### get edge from filter
  raw_edge = cv.Canny(gray, 100, 200)
  edge = np.zeros((raw_edge.shape[0], raw_edge.shape[1]), dtype=np.uint8)

  ### make edge bold by using weight vale
  for i in range(edge.shape[0]):
    for j in range(edge.shape[1]):

        if(raw_edge[i][j] != 0):

          for p in range(-weight, weight):
            for q in range(-weight, weight):\

              new_edge_i = min(max(0, i + p), edge.shape[0] - 1)
              new_edge_j = min(max(0, j + q), edge.shape[1] - 1)
              edge[new_edge_i][new_edge_j] = 1

  return edge




def check_existence(NormalizedMask_npArr, RowID, ColID):
  """
  check whether the pixel is in the picture and white 
  """
  Height, Width = NormalizedMask_npArr.shape

  if(0 <= RowID and RowID <= Height-1 and 0 <= ColID and ColID <= Width-1):#if pixel is in mask image
    if(NormalizedMask_npArr[RowID][ColID]==1):#if pixel is white
      return True

    else:
      return False

  else:
    return False





def indicies(NormalizedMask_npArr):
  """
  @input: NormalizedMask_npArr
  @return: index of valid pixel
  """
  # height and width of mask image
  Height, Width = NormalizedMask_npArr.shape

  # index of omega area
  Omega = np.nonzero(NormalizedMask_npArr)
  RowIDArr = Omega[0]
  ColIDArr = Omega[1]
  RowIDvec = np.reshape(RowIDArr, [RowIDArr.shape[0], 1])
  ColIDvec = np.reshape(ColIDArr, [ColIDArr.shape[0], 1])
  OmegaCoordinateArr = np.concatenate([RowIDvec, ColIDvec], 1)

  ## 1. flag of neigbourhoods pixel.
  ## --> write TRUE if neigbourhoods pixel is exist, FALSE otherwise.
  ## 2. dictionary for omega's index
  Neighborhoods_existence = []
  NumberedOmega = np.zeros((Height, Width), dtype=np.int32)
  for i in range(OmegaCoordinateArr.shape[0]):

    # pixel coordination
    RowID, ColID = OmegaCoordinateArr[i]

    # record the existence of neighborhood 
    Neighborhoods_existence.append([check_existence(NormalizedMask_npArr, RowID, ColID+1),
                     check_existence(NormalizedMask_npArr, RowID, ColID-1),
                     check_existence(NormalizedMask_npArr, RowID+1, ColID),
                     check_existence(NormalizedMask_npArr, RowID-1, ColID),])

    ## numebr the Omega are(0,1,2....)
    NumberedOmega[RowID][ColID] = i

  return OmegaCoordinateArr, np.array(Neighborhoods_existence), NumberedOmega


def index4omega(omega, id_h, id_w):

  '''
  input: omega, point(id_h, id_w)
  return: index of point in omega
  '''

  p = np.array([id_h, id_w])
  match = np.all(omega==p, axis=1)
  index = np.where(match)[0][0]

  return index


def laplacian_at_coordinate_import(AChannelOf_NormalizedSrcImg, OmegaCoordinate,Neighborhood_existence):
  '''
  Function to calculate gradient with respect OmegaCoordinate
  return grad(source) with Dirichlet boundary condition
  '''
  # current cordinate
  RowID, ColID = OmegaCoordinate

  # calculate laplacian
  N = np.sum(Neighborhood_existence == True)
  val = (N * AChannelOf_NormalizedSrcImg[RowID, ColID]
         - (float(Neighborhood_existence[0]==True) * AChannelOf_NormalizedSrcImg[RowID, ColID+1])
         - (float(Neighborhood_existence[1]==True) * AChannelOf_NormalizedSrcImg[RowID, ColID-1])
         - (float(Neighborhood_existence[2]==True) * AChannelOf_NormalizedSrcImg[RowID+1, ColID])
         - (float(Neighborhood_existence[3]==True) * AChannelOf_NormalizedSrcImg[RowID-1, ColID]))

  return val


def laplacian_at_coordinate_mix(AChannelOf_NormalizedSrcImg, AChannelOf_NormalizedTargetImg, OmegaCoordinate, Neighborhood_existence):
  '''

  '''
  ## current coordinate
  RowID, ColID = OmegaCoordinate

  ## grad of source image
  Right_grad_srcImg = float(Neighborhood_existence[0]==True) * \
    (AChannelOf_NormalizedSrcImg[RowID, ColID] - AChannelOf_NormalizedSrcImg[RowID, ColID+1])
  Left_grad_srcImg = float(Neighborhood_existence[1]==True) * \
    (AChannelOf_NormalizedSrcImg[RowID, ColID] - AChannelOf_NormalizedSrcImg[RowID, ColID-1])
  Bottom_grad_srcImg = float(Neighborhood_existence[2]==True) * \
    (AChannelOf_NormalizedSrcImg[RowID, ColID] - AChannelOf_NormalizedSrcImg[RowID+1, ColID])
  Up_grad_srcImg = float(Neighborhood_existence[3]==True) * \
    (AChannelOf_NormalizedSrcImg[RowID, ColID] - AChannelOf_NormalizedSrcImg[RowID-1, ColID])

  ## grad of target image
  Right_grad_targetImg = float(Neighborhood_existence[0]==True) * \
    (AChannelOf_NormalizedTargetImg[RowID, ColID] - AChannelOf_NormalizedTargetImg[RowID, ColID+1])
  Left_grad_targetImg = float(Neighborhood_existence[1]==True) * \
    (AChannelOf_NormalizedTargetImg[RowID, ColID] - AChannelOf_NormalizedTargetImg[RowID, ColID-1])
  Bottom_grad_targetImg = float(Neighborhood_existence[2]==True) * \
    (AChannelOf_NormalizedTargetImg[RowID, ColID] - AChannelOf_NormalizedTargetImg[RowID+1, ColID])
  Up_grad_targetImg = float(Neighborhood_existence[3]==True) * \
    (AChannelOf_NormalizedTargetImg[RowID, ColID] - AChannelOf_NormalizedTargetImg[RowID-1, ColID])

  val = [0 for i in range(4)]
  val[0] = Right_grad_targetImg if abs(Right_grad_srcImg) < abs(Right_grad_targetImg) else Right_grad_srcImg
  val[1] = Left_grad_targetImg if abs(Left_grad_srcImg) < abs(Left_grad_targetImg) else Left_grad_srcImg
  val[2] = Bottom_grad_targetImg if abs(Bottom_grad_srcImg) < abs(Bottom_grad_targetImg) else Bottom_grad_srcImg
  val[3] = Up_grad_targetImg if abs(Up_grad_srcImg) < abs(Up_grad_targetImg) else Up_grad_srcImg

  return sum(val)





def coefficient_matrix(OmegaCoordinateArr, Neighborhood_existence, NumberedOmega, A, OmegaDim):
  '''
    Compute coefficient matrix A
  '''  
  for i in range(OmegaDim):

    ## progress bar
    progress_bar(i, OmegaDim-1)

    ## fill 4 or -1(Laplacian kernel)
    A[i, i] = 4
    RowID, ColID = OmegaCoordinateArr[i]

    ## right
    if(Neighborhood_existence[i][0]):
      j = NumberedOmega[RowID][ColID+1]
      A[i, j] = -1

    ## left
    if(Neighborhood_existence[i][1]):
      j = NumberedOmega[RowID][ColID-1]
      A[i, j] = -1

    ## bottom
    if(Neighborhood_existence[i][2]):
      j = NumberedOmega[RowID+1][ColID]
      A[i, j] = -1

    ## up
    if(Neighborhood_existence[i][3]):
      j = NumberedOmega[RowID-1][ColID]
      A[i, j] = -1

  return A


def importing_gradients(NormalizedSrcImg_npArr, NormalizedTargetImg_npArr, OmegaCoordinateArr, Outline, Neighborhood_existence, OmegaDim):
  '''
  compute matrix u (Ax = u)
  '''  
  ### output array
  u_R = np.zeros(OmegaDim)
  u_G = np.zeros(OmegaDim)
  u_B = np.zeros(OmegaDim)

  ### take laplacian
  for i in range(OmegaDim):

    ## progress
    progress_bar(i, OmegaCoordinateArr.shape[0]-1)

    ## apply each color channel
    u_R[i] = laplacian_at_coordinate_import(NormalizedSrcImg_npArr[:, :, 2], OmegaCoordinateArr[i], Neighborhood_existence[i]) \
                  + constrain(NormalizedTargetImg_npArr[:, :, 2], OmegaCoordinateArr[i], Outline, Neighborhood_existence[i])
    u_G[i] = laplacian_at_coordinate_import(NormalizedSrcImg_npArr[:, :, 1], OmegaCoordinateArr[i], Neighborhood_existence[i]) \
                  + constrain(NormalizedTargetImg_npArr[:, :, 1], OmegaCoordinateArr[i], Outline, Neighborhood_existence[i])
    u_B[i] = laplacian_at_coordinate_import(NormalizedSrcImg_npArr[:, :, 0], OmegaCoordinateArr[i], Neighborhood_existence[i]) \
                  + constrain(NormalizedTargetImg_npArr[:, :, 0], OmegaCoordinateArr[i], Outline, Neighborhood_existence[i])

  return  u_R , u_G, u_B


def mixing_gradients(NormalizedSrcImg_npArr, NormalizedTargetImg_npArr, OmegaCoordinateArr, Outline, Neighborhood_existence, OmegaDim):
  '''
    Create gradient matrix u
    return: laplacian(src)[channel]
  '''  
  ### array for 3 color channel
  u_R = np.zeros(OmegaDim)
  u_G = np.zeros(OmegaDim)
  u_B = np.zeros(OmegaDim)

  ### calculate laplacian
  for i in range(OmegaDim):

    ## progress
    progress_bar(i, OmegaDim-1)

    ## apply each color channel
    u_R[i] = laplacian_at_coordinate_mix(NormalizedSrcImg_npArr[:, :, 2], NormalizedTargetImg_npArr[:, :, 2], OmegaCoordinateArr[i], Neighborhood_existence[i]) \
                + constrain(NormalizedTargetImg_npArr[:, :, 2], OmegaCoordinateArr[i], Outline, Neighborhood_existence[i])
    u_G[i] = laplacian_at_coordinate_mix(NormalizedSrcImg_npArr[:, :, 1], NormalizedTargetImg_npArr[:, :, 1], OmegaCoordinateArr[i], Neighborhood_existence[i]) \
                + constrain(NormalizedTargetImg_npArr[:, :, 1], OmegaCoordinateArr[i], Outline, Neighborhood_existence[i])                  
    u_B[i] = laplacian_at_coordinate_mix(NormalizedSrcImg_npArr[:, :, 0], NormalizedTargetImg_npArr[:, :, 0], OmegaCoordinateArr[i], Neighborhood_existence[i]) \
                + constrain(NormalizedTargetImg_npArr[:, :, 0], OmegaCoordinateArr[i], Outline, Neighborhood_existence[i])
             
  return u_R, u_G, u_B


def constrain(AChannelOf_NormalizedTargetImg, OmegaCoordinate, Outline, Neighborhood_existence):
  '''
  Function to set grad(source) = target at boundary
  return Dirichlet boundary condition for Coordinate
  '''
  ## current coordinate
  RowID, ColID = OmegaCoordinate

  ## In order to use "Dirichlet boundary condition",
  ## if on boundry, add in target intensity --> set constraint grad(source) = target at boundary
  if(Outline[RowID][ColID]==1):#if current pixel is on boundary
    val = (float(Neighborhood_existence[0]==False) * AChannelOf_NormalizedTargetImg[RowID, ColID+1]#right
           + float(Neighborhood_existence[1]==False) * AChannelOf_NormalizedTargetImg[RowID, ColID-1]#left
           + float(Neighborhood_existence[2]==False) * AChannelOf_NormalizedTargetImg[RowID+1, ColID]#bottom
           + float(Neighborhood_existence[3]==False) * AChannelOf_NormalizedTargetImg[RowID-1, ColID])#up
    return val

  ## If not on boundry, just take laplacian.
  else:#Outline[RowID][ColID]==0
    val = 0.0
    return val


def progress_bar(i, Ndim):
  """
  print progress bar
  """

  Step = 2#percent 每增加2才加一个#
  Percent = float(i) / float(Ndim) * 100

  ## convert percent to bar
  Current = "#" * int(Percent//Step)
  Remaining = " " * int(100/Step-int(Percent//Step))
  Progress_bar = "|{}{}|".format(Current, Remaining)
  print("\r{}:{:3.0f}[%]".format(Progress_bar, Percent), end="", flush=True)#\r将光标移到行首，但是不会移到下一行，如果继续输入的话会覆盖掉前面的内容
  

def poisson_import(NormalizedSrcImg_npArr, NormalizedMask_npArr, NormalizedTargetImg_npArr, GuidanceField, Result_dir):
  """
    compute Au = b
    ------------------------------------------------------------------------------------
     A: Coefficient matrix
     u: target ROI's pixel value
     b: The Laplacian of source img's interior and the function value of target img's boundary
  """
  # extract the outline of mask image(theta omega)
  Outline = get_outline(NormalizedMask_npArr) 
  NormalizedMask_npArr = np.array(NormalizedMask_npArr, dtype=np.uint8)# float64->uint8

  # get the coordination of pixles in omega area, the neighborhood's existence of omega pixels; and Number the pixels in omega
  OmegaCoordinateArr, Neighborhood_existence, NumberedOmega = indicies(NormalizedMask_npArr)

  # compute coefficient matrix A
  print("1. compute coefficient matrix A")
  OmegaDim = OmegaCoordinateArr.shape[0]
  A = sp.lil_matrix((OmegaDim, OmegaDim), dtype=np.float32) #list of list

  if(os.path.isfile("{0}/A.mat".format(Result_dir))):
    A = scipy.io.loadmat("{}/A".format(Result_dir))["A"]
    print("load coefficient matrix: A from .mat file\n")
  else:
    A = coefficient_matrix(OmegaCoordinateArr, Neighborhood_existence, NumberedOmega, A, OmegaDim)
    scipy.io.savemat("{}/A".format(Result_dir), {"A":A}) 
    print("\n")

  ### fill u
  ### --> each color channel
  print("step2: filling gradient matrix: b")

  u_R = np.zeros(OmegaDim)#red channel
  u_G = np.zeros(OmegaDim)#green channel
  u_B = np.zeros(OmegaDim)#blue channel
  ## select process type
  if(GuidanceField == "import"):
    u_R, u_G, u_B = importing_gradients(NormalizedSrcImg_npArr, NormalizedTargetImg_npArr, OmegaCoordinateArr, Outline, Neighborhood_existence, OmegaDim)
    print("\n")
  if(GuidanceField == "mix"):
    u_R, u_G, u_B =  mixing_gradients(NormalizedSrcImg_npArr, NormalizedTargetImg_npArr, OmegaCoordinateArr, Outline, Neighborhood_existence, OmegaDim)
    print("\n")

  ### solve
  print("step3: solve Ax = u")

  x_R, _ = sp.linalg.cg(A, u_R)
  x_G, _= sp.linalg.cg(A, u_G)
  x_B, _ = sp.linalg.cg(A, u_B)

  print("done!\n")

  ### create output by using x
  Seamless = NormalizedTargetImg_npArr.copy()
  Superimpose = NormalizedTargetImg_npArr.copy()

  for i in range(OmegaDim):

    RowID, ColID = OmegaCoordinateArr[i]
  
    ## seamless
    Seamless[RowID][ColID][0] = np.clip(x_B[i], 0.0, 1.0)
    Seamless[RowID][ColID][1] = np.clip(x_G[i], 0.0, 1.0)
    Seamless[RowID][ColID][2] = np.clip(x_R[i], 0.0, 1.0)

    ## superipose
    Superimpose[RowID][ColID][0] = NormalizedSrcImg_npArr[RowID][ColID][0]
    Superimpose[RowID][ColID][1] = NormalizedSrcImg_npArr[RowID][ColID][1]
    Superimpose[RowID][ColID][2] = NormalizedSrcImg_npArr[RowID][ColID][2]

  #从normalized复原
  return (np.array(Seamless*255, dtype=np.uint8), 
          np.array(Superimpose*255, dtype=np.uint8))
