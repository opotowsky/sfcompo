'''
sfcompo.py

Kenneth Dayman -- Oak Ridge National Laboratory -- Sept 2017

Module for working with SFCompo 2.0 Database in Pandas DataFrame form. Includes
tools for separating values from units, converting units, scrubbing SFCompo
*.csv files, and specialized searching on sorting utilities on top of native
Pandas functionality.

Included Functions:

1) load:         Loads the current version of the data from a pickle
2) scrub:        Executes scrubbing functions (i.e., runs commands from Jupyter
                 notebook (SFCompo_Munging.ipynb) using the defaults/values as
                 of Sept 2017. Running the notebook is preferable since small
                 particulars of the data and munging needs are likely to need
                 tweaking as the SFCompo web-based database is updated.
3) parindex:     Looks for partial matches in the dF index
4) getUnit:      Separates value from unit and returns a numeric value and
                 string unit
5) extractUnits: Gets units for each entry (row x column) in dF and generates
                 separate dF for use in convertUnit
6) convertUnit:  Finds the unit of given entry, and converts the associated
                 value to the requested unit. If the conversion factor is not
                 know, the user is prompted for the appropriate conversion
                 factor, which is the then saved for future use.

'''
# ----------------- #
# import statements #
# ----------------- #

import numpy as np
import pandas as pd
import re as re
import glob as glob
import pickle

# --------- #
# functions #
# --------- #

def setDataPath(path):
    global __DATAPATH__
    if __DATAPATH__ is None:
        __DATAPATH__ = path
    else:
        raise RuntimeError("Data path has already been set.")


def load():
    '''
    Loads SFCompo data and conversion factor table.
    '''

    if __DATAPATH__ is None:
        raise RuntimeError("Data path has not been set.")

    local_pickles = glob.glob(__DATAPATH__ + '/*.pkl')
    for p in local_pickles:
        fields = p.split('/')
        if 'SFCompo' in fields[-1]:
            X = np.load(p)
            break

    with open(__DATAPATH__ + '/U.pkl', 'rb') as handle:
        U = pickle.load(handle)

    with open(__DATAPATH__ + '/cF.pkl', 'rb') as handle:
        cF = pickle.load(handle)

    return X, U, cF


def scrub():
    pass


def getUnit(s):
    '''
    Given a string, finds the numeric value and remainder, presumably a unit
    indication. Returns a float and a string. If no numberic is found, two NaN
    are returned.
    '''
    if isinstance(s, basestring):
        # separate unit and value
        # trying with \A option such that the numerics should start the string
        queryString = '\A' + '\d*\,*\d*\.*\d+e*[+,-]*\d*' + '[-,x]*' + '\d*\,*\d*\.*\d*e*[+,-]*\d*'
        value  = re.findall(queryString, s)
        # if the entry is non-numeric (e.g., 'PWR')
        if len(value) == 0:
            return s, np.nan
        # there is one block of numerics in the string
        elif len(value) == 1:
            # there is only the numeric portion
            if len(s) == len(value[0]):
                try:
                    numeric = float(value[0].replace(',', ''))
                except:
                    numeric = value[0]
                return numeric, np.nan
            # there is additional stuff in the string other than the the numeric
            else:
                if ' ' in s:
                    numeric = float(value[0].replace(',', ''))
                    unit = s[len(value[0]):].strip()
                    return numeric, unit
                else:
                    return s, np.nan
        # there are multiple numeric portions in the string
        else:
            if ' ' in s:
                numeric = float(value[0].replace(',', ''))
                unit = s[len(value[0]):].strip()
                return numeric, unit
            else:
                return s, np.nan
    else:
        # if the entry is a float (without unit), return value and escape
        # this happens for dimensionless numbers like A or Z
        if isinstance(s, float) or isinstance(s, int):
            return s, np.nan

        # if NaN is passed, return and escape
        if np.isnan(s):
            return np.nan, np.nan


def parindex(X, q):
    '''
    Returns a subset of dF X such that query q is contained in index entries
    '''
    return X[X.index.map(lambda x: q in x)]


def initializeConversionFactors():
    cFactors = {'mm': {'cm': 1/10., 'm': 1/1000., 'in': 0.0393701},
                'cm': {'mm': 10., 'm': 1/100., 'in': 0.393701},
                'm': {'cm': 100., 'mm': 1000., 'in': 39.3701},
                'in': {'cm': 2.54, 'm': 0.0254, 'mm': 25.4},
                'g': {'mg': 1000., 'kg': 1/1000., 'lb': 0.00220462},
                'mg': {'g': 1e-3, 'kg': 1e-6, 'lb': 2.20462e-6},
                'kg': {'mg': 1e6, 'g': 1e3, 'lb': 2.20462},
                'lb': {'mg': 453592., 'kg': 0.453592, 'g': 453.592},
                'psi': {'bar': 0.0689476, 'Pa': 689476.},
                'bar': {'psi': 14.5038, 'Pa': 1e5},
                'Pa': {'bar': 1e-5, 'psi': 0.000145038},
                'MW*d/kgUi': {'MW*d/tUi': 1000.},
                'MW*d/tUi': {'MW*d/kgUi': 1/1000.}}
    return cFactors


def askCF(msg):
    print msg
    factor = raw_input('Please enter correct conversion factor: ')
    try:
        return float(factor)
    except:
        raise RuntimeError("Could not convert user-supplied conversion factor to numeric.")


def getCF(start, end, cF):
    # check start is in the dictionary
    if start not in cF.keys():
        msg = "No conversion factors for given starting unit %s" % (start)
        factor = askCF(msg)
        cF[start] = {end: factor}
        saveCFTable(cF)
        return float(factor)
    # have some data for the starting unit, so a dict exists, but not desired
    # end unit
    if end not in cF[start].keys():
        # see if reciprocal conversion factor is in the table
        try:
            factor = float(1/cF[end][start])
            cF[start][end] = factor
            saveCFTable(cF)
            return float(1/cF[end][start])
        # reciprocal not in table, ask user for conversion and save it
        except:
            msg = "No conversion factors for %s to %s." % (start, end)
            factor = askCF(msg)
            cF[start][end] = factor
            saveCFTable(cF)
            return float(factor)
    return cF[start][end]


def saveCFTable(cF):
    with open(__DATAPATH__ + '/cF.pkl', 'wb') as handle:
        pickle.dump(cF, handle, protocol=pickle.HIGHEST_PROTOCOL)


def changeUnits_(row, rv, ru, u, cF):
    '''
    Internal function used by changeUnits under Pandas.apply()
    '''
    cur_unit = u[int(row['idx'])]
    cf = getCF(cur_unit, ru, cF)
    row[rv] = row[rv] * cf
    return row


def changeUnits(dF, requestedValue, requestedUnit, U, cF):
    '''
    Given an SFCompo dF, the requestedValue (column) is extracted, and all
    values in that column are converted to the requestedUnit before returning.
    ConversionFactor table is queried and user is prompted as necessary.
    '''
    idx_col = dF.columns.get_loc(requestedValue)
    u = U[:, idx_col]
    dF['idx'] = np.arange(dF.shape[0])
    X = dF[['idx', requestedValue]].copy()
    
    # apply the internal changeUnits_ function to each row. Lambda function is
    # used to accommodate the multiple arguments that changeUnits_ needs.
    
    def f(x):
        r = changeUnits_(x, requestedValue, requestedUnit, u, cF)
        return r

    # X = X.apply(f, axis=1)
    X = X.apply(lambda x: changeUnits_(x, requestedValue, requestedUnit, u, cF), axis=1)
    X = X.drop('idx', axis=1)
    return X


# ------------------------------------- #
# functions/processes to add to munging #
# ------------------------------------- #
'''
Note these functions should be added to the munging rountine. This is two parts:
(1) Add to the SFCompo_Munging.ipynb notebook and (2) Add to the *.scrub() 
routine when it's written
'''


def extractUnits(X):
    '''
    Iterates over dF X elementwise, extracting values and units.
    Units are saved into a separate nd.array that is returned
    and used later for doing unit conversions.
    '''
    u = []
    nR = X.shape[0]

    def f(x):
        value, unit = getUnit(x)
        u.append(unit)
        return value

    tmp = X.applymap(f)
    # reshape the list into the 2-d nd.array, note the first nR entries
    # are skipped because apply map goes through the first column twice
    U = np.reshape(u[nR:], newshape=X.shape, order='F')
    return tmp, U


def replaceNone(X):
    def f(x):
        if x is None or x == '-':
            return np.nan
        else:
            return x
    return X.applymap(f)


def dropIdentifiers(X):
    X = X.drop(['Assembly identifier', 'Rod identifier', 'Sample identifier'], axis=1)
    return X

# ----- #
# tests #
# ----- #

def test_getUnit():
    # set 1
    input1 = ['3.16 m', '1,234 mg/kg', '1,234.56 kg/m^3', '3.4e-5 m']
    ans_v = [3.16, 1234, 1234.56, 3.4e-5]
    ans_u = ['m', 'mg/kg', 'kg/m^3', 'm']
    for i, x in enumerate(input1):
        v, u = getUnit(x)
        assert(v == ans_v[i]), "Value for input %d failed" % i
        assert(u == ans_u[i]), "Unit for input %d failed" % i

    # set 2
    input2 = [123.4, 3.4e-5, '2', 3, '5.0e5','pwr','16-2','9x9','BE 14-16', '19C1B']
    ans_v = [123.4, 3.4e-5, 2, 3, 5.0e5, 'pwr', '16-2', '9x9', 'BE 14-16', '19C1B']
    for i, x in enumerate(input2):
        v, u = getUnit(x)
        assert(v == ans_v[i]), "Value for input %d failed"  % (i + 4)
        assert(np.isnan(u)), "Unit for input %d failed"  % (i + 4)

    # set 3
    v, u = getUnit(np.nan)
    assert(np.isnan(v)), "Value for input 14 failed"
    assert(np.isnan(u)), "Unit for input 14 failed"

    print "\tTests of function getUnit() passed."


def test_extractUnits():
    data = np.array([['Ben', 1, '14 cm', '1.2 kg', '1 kg/m^3'],
                     ['Jerry', 10, '15 m', '1.0e7 g', '2 g/cc'],
                     ['Alice', 100, '12.4 mm', '15.2 lb', '2.4 g/cm3']])
    col = ['Name', 'Z', 'Height','Weight','Density']
    dF = pd.DataFrame(data, columns=col)

    dF, U = extractUnits(dF)
    correct_units = np.array([[np.nan, np.nan, 'cm', 'kg', 'kg/m^3'],
                              [np.nan, np.nan, 'm', 'g', 'g/cc'],
                              [np.nan, np.nan, 'mm', 'lb', 'g/cm3']])
    correct_values = np.array([['Ben', 1.0, 14.0, 1.2, 1.0],
                               ['Jerry', 10.0, 15.0, 10000000.0, 2.0],
                               ['Alice', 100.0, 12.4, 15.2, 2.4]], dtype='O')
    assert((U == correct_units).all()), "One or more extract units do not match"
    assert((dF.values == correct_values).all()), "One or more reformatted entries in DataFrame are not correct"
    print "\tTests of function extractUnits() passed."


def runTests():
    print "Running Tests of SFCompo Module"
    test_getUnit()
    test_extractUnits()
    print "All tests passed."

# ------------------ #
# startup statements #
# ------------------ #

__DATAPATH__ = None

Dev_Mode = True
if Dev_Mode:
    try:
        runTests()
    except:
        print "Some kind of problem happended during testing."
