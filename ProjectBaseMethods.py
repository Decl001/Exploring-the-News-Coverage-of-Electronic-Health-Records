'''
This module contains methods that have
proved useful throughout the project
'''

def isNewHalfYear(date1, date2):
    try:
        date1 = date1.split('-')
        date1Year = int(date1[0])
        date1Month = int(date1[1])
        date2 = date2.split('-')
        date2Year = int(date2[0])
        date2Month = int(date2[1])
        if date1Year > date2Year:
            return True
        elif date1Month > 6 and date2Month <= 6:
            return True
        else:
            return False
    except:
        return None

def isNewHalfYearOffset(date1, date2, offset):
    '''
    This method adds an offset to the half year
    '''
    try:
        date1 = date1.split('-')
        date1Year = int(date1[0])
        date1Month = int(date1[1])
        date2 = date2.split('-')
        date2Year = int(date2[0])
        date2Month = int(date2[1])
        if date1Year > date2Year and date1Month > offset:
            return True
        elif date1Month > (6 + offset) and date2Month <= (6 + offset):
            return True
        elif date1Month > offset and date2Month < offset:
            return True
        else:
            return False
    except:
        return None
    
def isNewQuarter(date1,date2):
    try:
        date1 = date1.split('-')
        date1Year = int(date1[0])
        date1Month = int(date1[1])
        date2 = date2.split('-')
        date2Year = int(date2[0])
        date2Month = int(date2[1])
        if date1Year > date2Year:
            return True
        elif date1Month > 3 and date2Month < 3:
            return True
        elif date1Month > 6 and date2Month < 6:
            return True
        elif date1Month > 9 and date2Month < 9:
            return True
        else:
            return False
    except:
        return None

def generate_ticklabels(dataframe):
    tick_label_list = []
    start_date = dataframe.date[0]
    start_date = start_date.split('-')
    curr_year = int(start_date[0])
    curr_month = int(start_date[1])
    curr_pos = 0
    if curr_month < 7:
        curr_pos = 1
    else:
        curr_pos = 2
    tick_label_list.append(str(curr_year) + '-' + str(curr_pos))
    for i,date in enumerate(dataframe.date):
        if i != 0 and isNewHalfYear(date,dataframe.date[i-1]):
            if curr_pos == 1:
                curr_pos += 1
            else:
                curr_pos = 1
                curr_year += 1
            tick_label_list.append(str(curr_year) + '-' + str(curr_pos))
    return tick_label_list
