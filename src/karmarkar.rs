extern crate nalgebra as na;
use na::{
    base::allocator::Allocator, DefaultAllocator, DimMin, DimName, DimNameAdd, DimSub, MatrixN,
    VectorN, U1,
};

pub fn karmarkar<D>(
    c: &VectorN<f64, D>,
    amat: &MatrixN<f64, D>,
    b: &VectorN<f64, D>,
    x: &VectorN<f64, D>,
    gamma: f64,
    eps: f64,
    nloop: usize,
) -> Result<VectorN<f64, D>, &'static str>
where
    D: DimName + DimNameAdd<D> + DimMin<D>,
    <D as DimMin<D>>::Output: DimSub<U1>,
    DefaultAllocator: Allocator<f64, D>
        + Allocator<f64, D, D>
        + Allocator<f64, <D as DimMin<D>>::Output>
        + Allocator<f64, <D as DimMin<D>>::Output, D>
        + Allocator<f64, D, <D as DimMin<D>>::Output>
        + Allocator<f64, <<D as DimMin<D>>::Output as DimSub<U1>>::Output>
        + Allocator<f64, <D as DimMin<D>>::Output, <D as DimMin<D>>::Output>,
{
    let mut ans = x.clone();
    for _ in 0..nloop {
        let vk = b - amat * &ans;
        let mut vk2 = VectorN::<f64, D>::zeros();
        vk2.cmpy(1.0, &vk, &vk, 0.0);
        let ivk2 = MatrixN::<f64, D>::from_diagonal(&vk2)
            .try_inverse()
            .ok_or("Not found inverse.")?;
        let gmat = amat.transpose() * ivk2 * amat;
        let pgmat = gmat.pseudo_inverse(1.0e-9)?;
        let d = pgmat * c;
        if d.norm() < eps {
            break;
        }
        let hv = -amat * &d;
        if hv.amax() <= 0.0 {
            return Err("Unbounded!");
        }
        let sa = (0..hv.nrows())
            .filter_map(|i| {
                if hv[i] > 0.0 {
                    Some(vk[i] / hv[i])
                } else {
                    None
                }
            })
            .fold(0.0 / 0.0, |m, v| v.min(m));
        if sa.is_nan() {
            break;
        }
        let alpha = gamma * sa;
        ans -= alpha * &d;
    }
    Ok(ans)
}

#[test]
fn it_works() {
    use na::U2;
    let c = VectorN::<f64, U2>::from_vec(vec![-1.0, -1.0]);
    let amat = MatrixN::<f64, U2>::from_vec(vec![1.0, 1.0, 1.0, -1.0]);
    let b = VectorN::<f64, U2>::from_vec(vec![0.5, 1.0]);
    let x = VectorN::<f64, U2>::from_vec(vec![-2.0, -2.0]);
    let ans = karmarkar(&c, &amat, &b, &x, 0.5, 1.0e-3, 30);
    let expected = VectorN::<f64, U2>::from_vec(vec![0.23242187, 0.23242187]);
    println!("{:?}", ans);
    if let Ok(v) = ans {
        const TORELANCE: f64 = 1.0e-6;
        assert!((v[0] - expected[0]).abs() < TORELANCE);
        assert!((v[1] - expected[1]).abs() < TORELANCE);
    }
}
