#just a template for now QUINDI NON RUNNARE PERCHÈ TANTO NON RUNNA

import numpy as np
from CSP_generics import Variable, Constraint

#the user is gonna give me the upper and lower investment limit, maximum risk
upper_limit = 3.0
lower_limit = 2.0
maximum_risk = 0.7
#the variable gone be the 10 assets, and the solution is gonna be how much to invest in each asset
domain = np.arange(0, upper_limit + 0.01, 0.01)

def asset_constrainst(p1):
    #made up method that's gonna do some ML on it lol and return a dictionary with some values
    values = p1.execute()
    return (values.volatilty <= maximum_risk)

def total_investments(p1, p2, p3):
    total = p1 + p2 + p3
    #and total expected revenue >= desired revenue???
    return (total >= lower_limit & total <= upper_limit)

AAPL = Variable('AAPL.csv', domain )
AMZN = Variable('AMZN.csv', domain )
STBKS = Variable('STBKS.csv', domain )
#[...]

constraints = []

constraints.append(Constraint(AAPL, asset_constrainst(AAPL)))
#constraints.append(... )...
