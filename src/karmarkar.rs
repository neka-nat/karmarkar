extern crate nalgebra as na;
use na::{
    base::allocator::Allocator, DefaultAllocator, DimMin, DimName, DimNameAdd, DimSub, MatrixN,
    RealField, VectorN, U1,
};
use num_traits::Float;

/// max c.T * x, subj A * x < b
pub fn karmarkar<F, D>(
    c: &VectorN<F, D>,
    amat: &MatrixN<F, D>,
    b: &VectorN<F, D>,
    x: &VectorN<F, D>,
    gamma: F,
    eps: F,
    nloop: usize,
) -> Result<VectorN<F, D>, &'static str>
where
    F: RealField + Float,
    D: DimName + DimNameAdd<D> + DimMin<D>,
    <D as DimMin<D>>::Output: DimSub<U1>,
    DefaultAllocator: Allocator<F, D>
        + Allocator<F, D, D>
        + Allocator<F, <D as DimMin<D>>::Output>
        + Allocator<F, <D as DimMin<D>>::Output, D>
        + Allocator<F, D, <D as DimMin<D>>::Output>
        + Allocator<F, <<D as DimMin<D>>::Output as DimSub<U1>>::Output>
        + Allocator<F, <D as DimMin<D>>::Output, <D as DimMin<D>>::Output>,
{
    let mut ans = x.clone();
    let tol = F::from_f32(1.0e-9).ok_or("Fail to convert f32 to F.")?;
    for _ in 0..nloop {
        let vk = b - amat * &ans;
        let mut vk2 = VectorN::<F, D>::zeros();
        vk2.cmpy(F::one(), &vk, &vk, F::zero());
        let ivk2 = MatrixN::<F, D>::from_diagonal(&VectorN::<F, D>::identity().component_div(&vk2));
        let gmat = amat.transpose() * ivk2 * amat;
        let pgmat = gmat.pseudo_inverse(tol)?;
        let d = pgmat * c;
        if d.norm() < eps {
            break;
        }
        let hv = -amat * &d;
        if hv.amax() <= F::zero() {
            return Err("Unbounded!");
        }
        let sa = (0..hv.nrows())
            .filter_map(|i| {
                if hv[i] > F::zero() {
                    Some(vk[i] / hv[i])
                } else {
                    None
                }
            })
            .fold(Float::infinity(), |m, v| Float::min(v, m));
        if !sa.is_finite() {
            break;
        }
        let alpha = gamma * sa;
        ans -= d * alpha;
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
