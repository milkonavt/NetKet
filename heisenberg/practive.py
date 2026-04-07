import numpy as np
class Employee():
    def __init__(self, first_name, last_name):
        self.first_name = first_name
        self.last_name = last_name

    def full_name(self):
        return self.first_name + ' ' + self.last_name


class cmt_division(Employee):
    def __init__(self, first_name, last_name, year):
        super().__init__(first_name, last_name)   # 🔑 call parent
        self.year = year

    def full_bio(self,a):
        return self.first_name + ' ' + self.last_name + ' ' + str(self.year)+' ' +str( a)


e1 = Employee('alex', 'nikolaenko')
print(e1.full_name())

e2 = cmt_division('alex', 'nikolaenko', 5)
print(e2.full_bio(4))

for (a,b) in [(4,5),(5,6)]:
    print(a,' ',b,' ',a+b)


PSx = np.array([
    [[10, 11], [12, 13]], # Layer 0 (index 0)
    [[20, 21], [22, 23]]  # Layer 1 (index 1)
])
print(PSx.reshape(PSx.shape[0], -1))

# #this is incorrect
# hi_f= nk.hilbert.SpinOrbitalFermions(
#     N, s=1 / 2, n_fermions_per_spin=n_fermions_per_spin
# )
# hi_s = nk.hilbert.Spin(s=1 / 2, N=N, total_sz=0)
# hi = nk.hilbert.TensorHilbert(hi_f,hi_s)

class TotalSzZeroConstraint(DiscreteHilbertConstraint):
    def __call__(self, x):
        # x shape: (..., total_number_of_dofs)

        # Adjust these slices to your actual TensorHilbert ordering
        # For SpinOrbitalFermions with spin-1/2, the fermionic part is occupations 0/1.
        # Suppose first 2*N entries are fermions, last N are local spins.
        down = x[:, :self.n_sites]  # first N entries

        print(down)
        up = x[:, self.n_sites:2 * self.n_sites]  # last N entries

        print(up)
        spins = x[:, 2 * self.n_sites:]
        print(spins)


        # For nk.hilbert.Spin(s=1/2), local values are usually +/-1
        # so total Sz = 0.5 * sum(s)
        total_sz_twice = jnp.sum(up, axis=-1)-jnp.sum(down, axis=-1) + jnp.sum(spins, axis=-1)

        return total_sz_twice == 0