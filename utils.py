from datetime import datetime
from dateutil.relativedelta import relativedelta

format = "%Y-%m-%d"

def add_time(starting, day=0, month=0, year=0):
    """
    Is a pain to add time to strings so this function just simplifies a little bit by converting the str to a date time and then adding on and then coverting back
    :param starting: The date you want to add to
    :param day: the number of days you want to add
    :param month: the number of months you want to add
    :param year: the number of years you want to add
    :return: The new date
    """
    return (datetime.strptime(starting, format)
            + relativedelta(months=month, days=day, years=year)).strftime(format)

def subtract_time(starting, day=0, month=0, year=0):

    """
    Identical to above but subtracting, see above.
    :param starting:
    :param day:
    :param month:
    :param year:
    :return:
    """
    return (datetime.strptime(starting, format)
            - relativedelta(months=month, days=day, years=year)).strftime(format)

def calc_diff(starting, ending, type='days'):
    """
    Calcualtes the number of days between 2 dates
    :param starting: start date
    :param ending: end date
    :return: the difference between the dates in days
    """
    # Note the cheeky minus at the beginning
    if type == 'days':
        return (datetime.strptime(ending, format) - datetime.strptime(starting, format)).days
    if type == 'weeks':
        return round((datetime.strptime(ending, format) - datetime.strptime(starting, format)).days / 7)
    if type == 'months':
        return (datetime.strptime(ending, format).year - datetime.strptime(starting, format).year) * 12 \
               + (datetime.strptime(ending, format).month - datetime.strptime(starting, format).month)

