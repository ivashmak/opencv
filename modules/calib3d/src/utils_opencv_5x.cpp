#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/types_c.h>

#include "utils_opencv_5x.hpp"

namespace cv {
void projectPoints5x( InputArray _objectPoints,
                        InputArray _rvec, InputArray _tvec,
                        InputArray _cameraMatrix, InputArray _distCoeffs,
                        OutputArray _imagePoints, OutputArray _dpdr,
                        OutputArray _dpdt, OutputArray _dpdf,
                        OutputArray _dpdc, OutputArray _dpdk,
                        OutputArray _dpdo, double aspectRatio)
{
    Mat _m, objectPoints = _objectPoints.getMat();
    Mat dpdr, dpdt, dpdc, dpdf, dpdk, dpdo;

    int i, j;
    double R[9], dRdr[27], t[3], a[9], k[14] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0}, fx, fy, cx, cy;
    Matx33d matTilt = Matx33d::eye();
    Matx33d dMatTiltdTauX(0,0,0,0,0,0,0,-1,0);
    Matx33d dMatTiltdTauY(0,0,0,0,0,0,1,0,0);
    Mat matR( 3, 3, CV_64F, R ), _dRdr( 3, 9, CV_64F, dRdr );
    double *dpdr_p = 0, *dpdt_p = 0, *dpdk_p = 0, *dpdf_p = 0, *dpdc_p = 0;
    double* dpdo_p = 0;
    int dpdr_step = 0, dpdt_step = 0, dpdk_step = 0, dpdf_step = 0, dpdc_step = 0;
    int dpdo_step = 0;
    bool fixedAspectRatio = aspectRatio > FLT_EPSILON;

    int objpt_depth = objectPoints.depth();
    int objpt_cn = objectPoints.channels();
    int total = (int)(objectPoints.total()*objectPoints.channels());
    int count = total / 3;
    if(total % 3 != 0)
    {
        //we have stopped support of homogeneous coordinates because it cause ambiguity in interpretation of the input data
        CV_Error( CV_StsBadArg, "Homogeneous coordinates are not supported" );
    }
    count = total / 3;
    CV_Assert(objpt_depth == CV_32F || objpt_depth == CV_64F);
    CV_Assert((objectPoints.rows == 1 && objpt_cn == 3) ||
              (objectPoints.rows == count && objpt_cn*objectPoints.cols == 3) ||
              (objectPoints.rows == 3 && objpt_cn == 1 && objectPoints.cols == count));

    Mat matM(objectPoints.size(), CV_64FC(objpt_cn));
    objectPoints.convertTo(matM, CV_64F);
    if (objectPoints.rows == 3 && objectPoints.cols == count) {
        Mat temp;
        transpose(matM, temp);
        matM = temp;
    }

    CV_Assert( _imagePoints.needed() );
    _imagePoints.create(count, 1, CV_MAKETYPE(objpt_depth, 2), -1, true);
    Mat ipoints = _imagePoints.getMat();
    ipoints.convertTo(_m, CV_64F);
    const Point3d* M = matM.ptr<Point3d>();
    Point2d* m = _m.ptr<Point2d>();

    Mat rvec = _rvec.getMat(), tvec = _tvec.getMat();
    if(!((rvec.depth() == CV_32F || rvec.depth() == CV_64F) &&
        (rvec.size() == Size(3, 3) ||
        (rvec.rows == 1 && rvec.cols*rvec.channels() == 3) ||
        (rvec.rows == 3 && rvec.cols*rvec.channels() == 1)))) {
        CV_Error(CV_StsBadArg, "rvec must be 3x3 or 1x3 or 3x1 floating-point array");
    }

    if( rvec.size() == Size(3, 3) )
    {
        rvec.convertTo(matR, CV_64F);
        Vec3d rvec_d;
        Rodrigues(matR, rvec_d);
        Rodrigues(rvec_d, matR, _dRdr);
        rvec.convertTo(matR, CV_64F);
    }
    else
    {
        double r[3];
        Mat _r(rvec.size(), CV_64FC(rvec.channels()), r);
        rvec.convertTo(_r, CV_64F);
        Rodrigues(_r, matR, _dRdr);
    }

    if(!((tvec.depth() == CV_32F || tvec.depth() == CV_64F) &&
        ((tvec.rows == 1 && tvec.cols*tvec.channels() == 3) ||
        (tvec.rows == 3 && tvec.cols*tvec.channels() == 1)))) {
        CV_Error(CV_StsBadArg, "tvec must be 1x3 or 3x1 floating-point array");
    }

    Mat _t(tvec.size(), CV_64FC(tvec.channels()), t);
    tvec.convertTo(_t, CV_64F);

    Mat cameraMatrix = _cameraMatrix.getMat();

    if(cameraMatrix.size() != Size(3, 3) || cameraMatrix.channels() != 1)
        CV_Error( CV_StsBadArg, "Intrinsic parameters must be 3x3 floating-point matrix" );
    Mat _a(3, 3, CV_64F, a);
    cameraMatrix.convertTo(_a, CV_64F);

    fx = a[0]; fy = a[4];
    cx = a[2]; cy = a[5];

    if( fixedAspectRatio )
        fx = fy*aspectRatio;

    Mat distCoeffs = _distCoeffs.getMat();
    int ktotal = 0;
    if( distCoeffs.data )
    {
        int kcn = distCoeffs.channels();
        ktotal = (int)distCoeffs.total()*kcn;
        if( (distCoeffs.rows != 1 && distCoeffs.cols != 1) ||
            (ktotal != 4 && ktotal != 5 && ktotal != 8 && ktotal != 12 && ktotal != 14))
            CV_Error( CV_StsBadArg, "Distortion coefficients must be 1x4, 4x1, 1x5, 5x1, 1x8, 8x1, 1x12, 12x1, 1x14 or 14x1 floating-point vector" );

        Mat _k(distCoeffs.size(), CV_64FC(kcn), k);
        distCoeffs.convertTo(_k, CV_64F);
        if(k[12] != 0 || k[13] != 0)
            computeTiltProjectionMatrix(k[12], k[13], &matTilt, &dMatTiltdTauX, &dMatTiltdTauY);
    }

    if( _dpdr.needed() )
    {
        dpdr.create(count*2, 3, CV_64F);
        dpdr_p = dpdr.ptr<double>();
        dpdr_step = (int)dpdr.step1();
    }
    if( _dpdt.needed() )
    {
        dpdt.create(count*2, 3, CV_64F);
        dpdt_p = dpdt.ptr<double>();
        dpdt_step = (int)dpdt.step1();
    }
    if( _dpdf.needed() )
    {
        dpdf.create(count*2, 2, CV_64F);
        dpdf_p = dpdf.ptr<double>();
        dpdf_step = (int)dpdf.step1();
    }
    if( _dpdc.needed() )
    {
        dpdc.create(count*2, 2, CV_64F);
        dpdc_p = dpdc.ptr<double>();
        dpdc_step = (int)dpdc.step1();
    }
    if( _dpdk.needed() )
    {
        dpdk.create(count*2, ktotal, CV_64F);
        dpdk_p = dpdk.ptr<double>();
        dpdk_step = (int)dpdk.step1();
    }
    if( _dpdo.needed() )
    {
        dpdo = Mat::zeros(count*2, count*3, CV_64F);
        dpdo_p = dpdo.ptr<double>();
        dpdo_step = (int)dpdo.step1();
    }

    bool calc_derivatives = dpdr.data || dpdt.data || dpdf.data ||
                            dpdc.data || dpdk.data || dpdo.data;

    for( i = 0; i < count; i++ )
    {
        double X = M[i].x, Y = M[i].y, Z = M[i].z;
        double x = R[0]*X + R[1]*Y + R[2]*Z + t[0];
        double y = R[3]*X + R[4]*Y + R[5]*Z + t[1];
        double z = R[6]*X + R[7]*Y + R[8]*Z + t[2];
        double r2, r4, r6, a1, a2, a3, cdist, icdist2;
        double xd, yd, xd0, yd0, invProj;
        Vec3d vecTilt;
        Vec3d dVecTilt;
        Matx22d dMatTilt;
        Vec2d dXdYd;

        double z0 = z;
        z = z ? 1./z : 1;
        x *= z; y *= z;

        r2 = x*x + y*y;
        r4 = r2*r2;
        r6 = r4*r2;
        a1 = 2*x*y;
        a2 = r2 + 2*x*x;
        a3 = r2 + 2*y*y;
        cdist = 1 + k[0]*r2 + k[1]*r4 + k[4]*r6;
        icdist2 = 1./(1 + k[5]*r2 + k[6]*r4 + k[7]*r6);
        xd0 = x*cdist*icdist2 + k[2]*a1 + k[3]*a2 + k[8]*r2+k[9]*r4;
        yd0 = y*cdist*icdist2 + k[2]*a3 + k[3]*a1 + k[10]*r2+k[11]*r4;

        // additional distortion by projecting onto a tilt plane
        vecTilt = matTilt*Vec3d(xd0, yd0, 1);
        invProj = vecTilt(2) ? 1./vecTilt(2) : 1;
        xd = invProj * vecTilt(0);
        yd = invProj * vecTilt(1);

        m[i].x = xd*fx + cx;
        m[i].y = yd*fy + cy;

        if( calc_derivatives )
        {
            if( dpdc.data )
            {
                dpdc_p[0] = 1; dpdc_p[1] = 0; // dp_xdc_x; dp_xdc_y
                dpdc_p[dpdc_step] = 0;
                dpdc_p[dpdc_step+1] = 1;
                dpdc_p += dpdc_step*2;
            }

            if( dpdf_p )
            {
                if( fixedAspectRatio )
                {
                    dpdf_p[0] = 0; dpdf_p[1] = xd*aspectRatio; // dp_xdf_x; dp_xdf_y
                    dpdf_p[dpdf_step] = 0;
                    dpdf_p[dpdf_step+1] = yd;
                }
                else
                {
                    dpdf_p[0] = xd; dpdf_p[1] = 0;
                    dpdf_p[dpdf_step] = 0;
                    dpdf_p[dpdf_step+1] = yd;
                }
                dpdf_p += dpdf_step*2;
            }
            for (int row = 0; row < 2; ++row)
                for (int col = 0; col < 2; ++col)
                    dMatTilt(row,col) = matTilt(row,col)*vecTilt(2) - matTilt(2,col)*vecTilt(row);
            double invProjSquare = (invProj*invProj);
            dMatTilt *= invProjSquare;
            if( dpdk_p )
            {
                dXdYd = dMatTilt*Vec2d(x*icdist2*r2, y*icdist2*r2);
                dpdk_p[0] = fx*dXdYd(0);
                dpdk_p[dpdk_step] = fy*dXdYd(1);
                dXdYd = dMatTilt*Vec2d(x*icdist2*r4, y*icdist2*r4);
                dpdk_p[1] = fx*dXdYd(0);
                dpdk_p[dpdk_step+1] = fy*dXdYd(1);
                if( dpdk.cols > 2 )
                {
                    dXdYd = dMatTilt*Vec2d(a1, a3);
                    dpdk_p[2] = fx*dXdYd(0);
                    dpdk_p[dpdk_step+2] = fy*dXdYd(1);
                    dXdYd = dMatTilt*Vec2d(a2, a1);
                    dpdk_p[3] = fx*dXdYd(0);
                    dpdk_p[dpdk_step+3] = fy*dXdYd(1);
                    if( dpdk.cols > 4 )
                    {
                        dXdYd = dMatTilt*Vec2d(x*icdist2*r6, y*icdist2*r6);
                        dpdk_p[4] = fx*dXdYd(0);
                        dpdk_p[dpdk_step+4] = fy*dXdYd(1);

                        if( dpdk.cols > 5 )
                        {
                            dXdYd = dMatTilt*Vec2d(
                              x*cdist*(-icdist2)*icdist2*r2, y*cdist*(-icdist2)*icdist2*r2);
                            dpdk_p[5] = fx*dXdYd(0);
                            dpdk_p[dpdk_step+5] = fy*dXdYd(1);
                            dXdYd = dMatTilt*Vec2d(
                              x*cdist*(-icdist2)*icdist2*r4, y*cdist*(-icdist2)*icdist2*r4);
                            dpdk_p[6] = fx*dXdYd(0);
                            dpdk_p[dpdk_step+6] = fy*dXdYd(1);
                            dXdYd = dMatTilt*Vec2d(
                              x*cdist*(-icdist2)*icdist2*r6, y*cdist*(-icdist2)*icdist2*r6);
                            dpdk_p[7] = fx*dXdYd(0);
                            dpdk_p[dpdk_step+7] = fy*dXdYd(1);
                            if( dpdk.cols > 8 )
                            {
                                dXdYd = dMatTilt*Vec2d(r2, 0);
                                dpdk_p[8] = fx*dXdYd(0); //s1
                                dpdk_p[dpdk_step+8] = fy*dXdYd(1); //s1
                                dXdYd = dMatTilt*Vec2d(r4, 0);
                                dpdk_p[9] = fx*dXdYd(0); //s2
                                dpdk_p[dpdk_step+9] = fy*dXdYd(1); //s2
                                dXdYd = dMatTilt*Vec2d(0, r2);
                                dpdk_p[10] = fx*dXdYd(0);//s3
                                dpdk_p[dpdk_step+10] = fy*dXdYd(1); //s3
                                dXdYd = dMatTilt*Vec2d(0, r4);
                                dpdk_p[11] = fx*dXdYd(0);//s4
                                dpdk_p[dpdk_step+11] = fy*dXdYd(1); //s4
                                if( dpdk.cols > 12 )
                                {
                                    dVecTilt = dMatTiltdTauX * Vec3d(xd0, yd0, 1);
                                    dpdk_p[12] = fx * invProjSquare * (
                                      dVecTilt(0) * vecTilt(2) - dVecTilt(2) * vecTilt(0));
                                    dpdk_p[dpdk_step+12] = fy*invProjSquare * (
                                      dVecTilt(1) * vecTilt(2) - dVecTilt(2) * vecTilt(1));
                                    dVecTilt = dMatTiltdTauY * Vec3d(xd0, yd0, 1);
                                    dpdk_p[13] = fx * invProjSquare * (
                                      dVecTilt(0) * vecTilt(2) - dVecTilt(2) * vecTilt(0));
                                    dpdk_p[dpdk_step+13] = fy * invProjSquare * (
                                      dVecTilt(1) * vecTilt(2) - dVecTilt(2) * vecTilt(1));
                                }
                            }
                        }
                    }
                }
                dpdk_p += dpdk_step*2;
            }

            if( dpdt_p )
            {
                double dxdt[] = { z, 0, -x*z }, dydt[] = { 0, z, -y*z };
                for( j = 0; j < 3; j++ )
                {
                    double dr2dt = 2*x*dxdt[j] + 2*y*dydt[j];
                    double dcdist_dt = k[0]*dr2dt + 2*k[1]*r2*dr2dt + 3*k[4]*r4*dr2dt;
                    double dicdist2_dt = -icdist2*icdist2*(k[5]*dr2dt + 2*k[6]*r2*dr2dt + 3*k[7]*r4*dr2dt);
                    double da1dt = 2*(x*dydt[j] + y*dxdt[j]);
                    double dmxdt = (dxdt[j]*cdist*icdist2 + x*dcdist_dt*icdist2 + x*cdist*dicdist2_dt +
                                       k[2]*da1dt + k[3]*(dr2dt + 4*x*dxdt[j]) + k[8]*dr2dt + 2*r2*k[9]*dr2dt);
                    double dmydt = (dydt[j]*cdist*icdist2 + y*dcdist_dt*icdist2 + y*cdist*dicdist2_dt +
                                       k[2]*(dr2dt + 4*y*dydt[j]) + k[3]*da1dt + k[10]*dr2dt + 2*r2*k[11]*dr2dt);
                    dXdYd = dMatTilt*Vec2d(dmxdt, dmydt);
                    dpdt_p[j] = fx*dXdYd(0);
                    dpdt_p[dpdt_step+j] = fy*dXdYd(1);
                }
                dpdt_p += dpdt_step*2;
            }

            if( dpdr_p )
            {
                double dx0dr[] =
                {
                    X*dRdr[0] + Y*dRdr[1] + Z*dRdr[2],
                    X*dRdr[9] + Y*dRdr[10] + Z*dRdr[11],
                    X*dRdr[18] + Y*dRdr[19] + Z*dRdr[20]
                };
                double dy0dr[] =
                {
                    X*dRdr[3] + Y*dRdr[4] + Z*dRdr[5],
                    X*dRdr[12] + Y*dRdr[13] + Z*dRdr[14],
                    X*dRdr[21] + Y*dRdr[22] + Z*dRdr[23]
                };
                double dz0dr[] =
                {
                    X*dRdr[6] + Y*dRdr[7] + Z*dRdr[8],
                    X*dRdr[15] + Y*dRdr[16] + Z*dRdr[17],
                    X*dRdr[24] + Y*dRdr[25] + Z*dRdr[26]
                };
                for( j = 0; j < 3; j++ )
                {
                    double dxdr = z*(dx0dr[j] - x*dz0dr[j]);
                    double dydr = z*(dy0dr[j] - y*dz0dr[j]);
                    double dr2dr = 2*x*dxdr + 2*y*dydr;
                    double dcdist_dr = (k[0] + 2*k[1]*r2 + 3*k[4]*r4)*dr2dr;
                    double dicdist2_dr = -icdist2*icdist2*(k[5] + 2*k[6]*r2 + 3*k[7]*r4)*dr2dr;
                    double da1dr = 2*(x*dydr + y*dxdr);
                    double dmxdr = (dxdr*cdist*icdist2 + x*dcdist_dr*icdist2 + x*cdist*dicdist2_dr +
                                       k[2]*da1dr + k[3]*(dr2dr + 4*x*dxdr) + (k[8] + 2*r2*k[9])*dr2dr);
                    double dmydr = (dydr*cdist*icdist2 + y*dcdist_dr*icdist2 + y*cdist*dicdist2_dr +
                                       k[2]*(dr2dr + 4*y*dydr) + k[3]*da1dr + (k[10] + 2*r2*k[11])*dr2dr);
                    dXdYd = dMatTilt*Vec2d(dmxdr, dmydr);
                    dpdr_p[j] = fx*dXdYd(0);
                    dpdr_p[dpdr_step+j] = fy*dXdYd(1);
                }
                dpdr_p += dpdr_step*2;
            }

            if( dpdo_p )
            {
                double dxdo[] = { z * ( R[0] - x * z * z0 * R[6] ),
                                  z * ( R[1] - x * z * z0 * R[7] ),
                                  z * ( R[2] - x * z * z0 * R[8] ) };
                double dydo[] = { z * ( R[3] - y * z * z0 * R[6] ),
                                  z * ( R[4] - y * z * z0 * R[7] ),
                                  z * ( R[5] - y * z * z0 * R[8] ) };
                for( j = 0; j < 3; j++ )
                {
                    double dr2do = 2 * x * dxdo[j] + 2 * y * dydo[j];
                    double dr4do = 2 * r2 * dr2do;
                    double dr6do = 3 * r4 * dr2do;
                    double da1do = 2 * y * dxdo[j] + 2 * x * dydo[j];
                    double da2do = dr2do + 4 * x * dxdo[j];
                    double da3do = dr2do + 4 * y * dydo[j];
                    double dcdist_do
                        = k[0] * dr2do + k[1] * dr4do + k[4] * dr6do;
                    double dicdist2_do = -icdist2 * icdist2
                        * ( k[5] * dr2do + k[6] * dr4do + k[7] * dr6do );
                    double dxd0_do = cdist * icdist2 * dxdo[j]
                        + x * icdist2 * dcdist_do + x * cdist * dicdist2_do
                        + k[2] * da1do + k[3] * da2do + k[8] * dr2do
                        + k[9] * dr4do;
                    double dyd0_do = cdist * icdist2 * dydo[j]
                        + y * icdist2 * dcdist_do + y * cdist * dicdist2_do
                        + k[2] * da3do + k[3] * da1do + k[10] * dr2do
                        + k[11] * dr4do;
                    dXdYd = dMatTilt * Vec2d( dxd0_do, dyd0_do );
                    dpdo_p[i * 3 + j] = fx * dXdYd( 0 );
                    dpdo_p[dpdo_step + i * 3 + j] = fy * dXdYd( 1 );
                }
                dpdo_p += dpdo_step * 2;
            }
        }
    }

    _m.convertTo(_imagePoints, objpt_depth);

    int depth = CV_64F;//cameraMatrix.depth();
    if( _dpdr.needed() )
        dpdr.convertTo(_dpdr, depth);

    if( _dpdt.needed() )
        dpdt.convertTo(_dpdt, depth);

    if( _dpdf.needed() )
        dpdf.convertTo(_dpdf, depth);

    if( _dpdc.needed() )
        dpdc.convertTo(_dpdc, depth);

    if( _dpdk.needed() )
        dpdk.convertTo(_dpdk, depth);

    if( _dpdo.needed() )
        dpdo.convertTo(_dpdo, depth);
}
}