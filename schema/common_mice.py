"""Schema for mouse related information"""

import datajoint as dj
from dateutil.parser import parse
import matplotlib.pyplot as plt
import numpy as np

schema = dj.schema('common_mice', locals(), create_tables=True)


@schema
class Investigator(dj.Lookup):
    definition = """    # Name and contact information of lab members
    username        : char(6)      # Unique username, should be the UZH shortname, has to be 6 characters long
    ---
    full_name       : varchar(128)    # First and last name
    email           : varchar(256)    # Contact email address
    """
    contents = [
        ['hheise', 'Hendrik Heiser', 'heiser@hifo.uzh.ch'],
        ['mpanze', 'Matteo Panzeri', 'panzeri@hifo.uzh.ch'],
        ['jnambi', 'Jithin Nambiar', 'nambiar@hifo.uzh.ch'],
        ['aswahl', 'Anna-Sophia Wahl', 'wahl@hifo.uzh.ch']
    ]


@schema
class Strain(dj.Lookup):
    definition = """    # Genetic type of the mouse
    strain          : varchar(128)    # Mouse variant short name
    ---
    formal_name     : varchar(2048)   # More detailled description of genetic background
    ca_indicator    : varchar(255)    # Expressed calcium indicator (e.g. GCaMP6f)
    layer           : varchar(255)    # Expression layer of indicator (e.g. L2/3 or all)
    """
    contents = [
        ['WT', 'C57BL/6J', 'None', 'None'],
        ['Snap25-RCaMP', 'Snap-Cre;tTA-C;RCaMP', 'RCaMP', 'all'],
        ['L2/3-RCaMP', 'L2/3Cre;tTA-C;RCaMP', 'RCaMP', 'L2/3'],
        ['L2/3-TIGRE1.0-GCaMP6f', 'L2/3Cre;tTA-C;GCa6f', 'GCaMP6f', 'L2/3'],
        ['L2/3-TIGRE2.0-GCaMP6f', 'Rasgrf2 x Ai148D', 'GCaMP6f', 'L2/3'],
        ['Thy1-GCaMP6f', 'C57BL/6J-Tg(Thy1-GCaMP6f)GP5.17Dkim/J', 'GCaMP6f', 'L2/3, L5, CA1, CA3, dentate gyrus'],
        ['Snap25-GCaMP6f', 'Snap-Cre;tTA-C;GCaP6 ', 'GCaMP6f', 'all'],
    ]


@schema
class Licence(dj.Lookup):
    definition = """    # Licence and project ID under which the mouse is kept
    licence_id      : varchar(128)      # Licence ID with project suffix to keep it unique
    ---
    -> Investigator                     # Link to licence holder 
    description     : varchar(512)      # Short description of the project   
    """
    # add current licence, retrieve the licence file from the server
    contents = [
       ['241/2018-A', 'aswahl', 'Studying sensorimotor recovery after stroke'],
       ['241/2018-B', 'aswahl', 'Studying cognitive impairment after stroke']
    ]


@schema
class Mouse(dj.Manual):
    definition = """ # Basic information about the Mouse
      -> Investigator                          # Link to investigator to which this mouse belongs
      mouse_id      : smallint                 # ID of mouse (unique per investigator)
      ---
      dob           : date                     # Day of birth (year-month-day)
      sex           : enum('M', 'F', 'U')      # Sex of mouse - Male, Female, or Unknown/Unclassified
      batch         : tinyint                  # Batch ID that this mouse belongs to (0 if no batch)
      -> Strain                                # Link to the genetic type of the mouse
      genotype      : enum('WT', 'n.d.', '+/+', '+/-', '-/-')  # n.d. if transgenic but not determined
      irats_id      : varchar(20)              # ID that is used in iRats (e.g. BJ5698)
      cage_num      : int                      # Cage number
      ear_mark      : varchar(10)              # Actual ear mark, might not correspond to iRats earmark
      -> Licence                               # Link to project ID
      info          : varchar(1024)            # Additional information about this mouse
      """

    def get_weight_threshold(self, printout=True, rel_threshold=0.85):
        """
        Give the pre-surgery weight threshold of a single Mouse query.
        :param printout: bool flag whether the result should be printed or returned as a float
        :param rel_threshold: float, optional; threshold of weight (default 85%)
        :return: rel_threshold fraction of the pre-surgery weight
        """
        mouse = self.fetch1()                                   # Get Mouse() entry (has to be single row)
        first_surg = Surgery() & mouse & "surgery_num=1"        # Filter for first surgery
        if len(first_surg) == 1:
            # Get date of first surgery and the weight from Weight() at that date
            first_surg_date = first_surg.fetch1('surgery_date').strftime("%Y-%m-%d")
            pre_op_weight = float((Weight() & mouse & "date_of_weight='{}'".format(first_surg_date)).fetch1('weight'))
            if printout:
                print("The {}% pre-surgery weight threshold of M{} is {:.1f}g.".format(rel_threshold, mouse['mouse_id'],
                                                                                       pre_op_weight * rel_threshold))
            else:
                return pre_op_weight*0.85
        else:
            if printout:
                print("M{} does not have a recorded surgery. {}% threshold cannot "
                      "be computed.".format(mouse['mouse_id'], rel_threshold))
            else:
                return None

    def plot_weight(self, relative=False, rel_threshold=0.85, show_surgeries=False):
        """
        Plots the weights across time of a Mouse query (one or many mice).
        :param relative: bool flag whether weights should be plotted as % of pre-surgery weight
        :param rel_threshold: float, optional; threshold of weight (default 85%)
        :param show_surgeries: bool float whether surgeries should be shown as vertical lines
        """
        mice = self.fetch('KEY', as_dict=True)          # Get primary keys of the current query for downstream querying

        ax = plt.subplot()                              # Initialize figure
        surg_list = []

        for mouse in mice:
            dates = (Weight & mouse).fetch('date_of_weight')    # Get list of dates and weights for the current mouse
            weights = (Weight & mouse).fetch('weight')
            # Get pre-surgery weight threshold of the current mouse
            thresh = float((self & mouse).get_weight_threshold(printout=False, rel_threshold=rel_threshold))

            if relative:
                # If relative weights should be plotted, normalize weights against threshold
                pre_surg = thresh / rel_threshold
                ax.plot(dates, np.array(weights, dtype=float)/pre_surg, label='M{}'.format(mouse['mouse_id']))
                ax.set_ylabel('weight [% of pre-surgery weight]')
            else:
                ax.plot(dates, weights, label='M{}'.format(mouse['mouse_id']))

            if show_surgeries:
                # Get surgeries from the current mouse and add any new surgeries to the list
                surg_date, surg_type = (Surgery & mouse).fetch('surgery_date', 'surgery_type')
                for curr_date, typ in zip(surg_date, surg_type):
                    curr_date = curr_date.date()            # remove time information to combine same-day surgeries
                    if (curr_date, typ) not in surg_list:
                        surg_list.append((curr_date, typ))

        # All unique surgeries of the plotted mice will be shown
        if show_surgeries:
            trans = ax.get_xaxis_transform()
            for curr_surg in surg_list:
                ax.axvline(curr_surg[0], color='g')
                ax.text(curr_surg[0], 0.99, curr_surg[1][:11]+'...', transform=trans, rotation=-90,
                        ha='left', va='top')

        if len(mice) == 1:
            if relative:
                ax.axhline(rel_threshold, color='r')
                ax.axhline(1, linestyle='--', color='gray', alpha=0.5)
                ax.set_title('M{} relative weight profile'.format(mice[0]['mouse_id']))
            else:
                ax.axhline(thresh, color='r')           # If only one mouse was plotted, draw threshold lines
                ax.set_title('M{} weight profile'.format(mice[0]['mouse_id']))
                ax.set_ylabel('weight [g]')
        else:
            if relative:
                ax.axhline(rel_threshold, color='r')
                ax.axhline(1, linestyle='--', color='gray', alpha=0.5)
                ax.set_title('Relative weight profiles')
                ax.legend()
            else:
                ax.set_ylabel('weight [g]')
                ax.set_title('Weight profiles')
                ax.legend()
        ax.set_xlabel('date')


@schema
class Weight(dj.Manual):
    definition = """ # Table that stores the weights of individual mice
    -> Mouse
    date_of_weight      : date           # Date of weighing (year-month-day)
    ---
    weight              : decimal(3,1)   # Weight in grams
    """

    def insert1(self, row, **kwargs):
        try:
            super().insert((row,), **kwargs)
        except dj.errors.DuplicateError:
            # Todo: Ask user if previous weight should be overwritten?
            print("A weight has already been recorded for M{} on {}.".format(row['mouse_id'], row['date_of_weight']))

        # Warn user if the 85% threshold of that mouse is crossed
        user_filt = "username='{}'".format(row['username'])
        mouse_filt = "mouse_id='{}'".format(row['mouse_id'])
        weight_thresh = (Mouse() & user_filt & mouse_filt).get_weight_threshold(printout=False)
        if (weight_thresh is not None) and (row['weight'] < weight_thresh):
            print("WARNING: The weight of M{} of {} is below the 85% pre-surgery threshold of {:.1f}!".format(
                row['mouse_id'], row['weight'], weight_thresh))

        # Todo: Warn user again if the 85% threshold has been crossed 3+ days ago without recovery


@schema
class CareSubstance(dj.Lookup):
    definition = """  # Different substances administered for pain management and post-OP care
    care_name       : varchar(128)           # Unique name of the substance
    ---
    role            : varchar(64)            # Type of substance
    dosage          : tinyint                # dosage in mg/kg body weight
    concentration   : tinyint                # concentration of injection in mg/mL
    """

    contents = [
        ['Carprofen (s.c.)', 'analgesic', 5, 20],
        ['Paracetamol (water)', 'analgesic', 0, 2],
        ['Baytril (s.c.)', 'antibiotic', 10, 1],
        ['Baytril (water)', 'antibiotic', 0, 0.1]
    ]


@schema
class PainManagement(dj.Manual):
    definition = """ # Pain management records, especially for post-OP care
    -> Mouse
    date_of_care        : date              # Date of care application (year-month-day)
    ---
    -> CareSubstance
    care_volume = 0     : tinyint           # volume of injection in uL (0 if administered through drinking water)
    care_frequency      : tinyint           # Number of administration per day (1 or 2)
    """


@schema
class Sacrificed(dj.Manual):
    definition = """ # Table to keep track of euthanized mice
    -> Mouse
    ---
    date_of_sacrifice   : date              # Date of sacrifice (year-month-day)
    perfused            : tinyint           # 0 for no, 1 for yes (brain fixed and kept)
    reason              : varchar(1024)     # Comments
    """


@schema
class Substance(dj.Lookup):
    definition = """  # Different substances that can be injected during a surgery
    substance_name  : varchar(128)      # Unique name of the substance
    ---
    full_name       : varchar(256)      # Long name of the substance
    type            : varchar(128)      # Type of substance
    supplier        : varchar(128)      # Supplier of the substance
    datasheet       : varchar(256)      # Website of substance with additional information
    """

    contents = [
       ['AAV9-hSyn-GCaMP6f', 'pAAV.Syn.GCaMP6f.WPRE.SV40', 'Viral calcium indicator', 'addgene',
        'https://www.addgene.org/100837/'],
       ['AAV2-CamKII-C1V1', 'pAAV-CaMKIIa-C1V1 (t/t)-TS-mCherry', 'Viral optogenetic stimulant', 'addgene',
        'https://www.addgene.org/35500/'],
       ['endothelin', 'endothelin 1 (ET-1)', 'vasoconstrictor', 'Sigma',
        'https://www.sigmaaldrich.com/catalog/product/sigma/e7764'],
       ['microspheres', 'Fluorescent PMMA Microparticles Red5 20 m', 'microspheres', 'PolyAn',
        'https://www.poly-an.de/micro-nanoparticles/fluorescent-pmma-microparticles/fluorescent-pmma-microparticles'],
       ['TMP', "Trimethoprim", "antibiotic", "Sigma", "https://www.sigmaaldrich.com/catalog/product/sigma/t7883"],
    ]


@schema
class SurgeryType(dj.Lookup):
    definition = """  # Different types of surgery
    surgery_type        : varchar(64)    # Description of surgery
    ---
    """
    contents = [
        ['Virus injection'],
        ['Head post'],
        ['Hippocampal window'],
        ['Motor cortex window'],
        ['Widefield preparation'],
        ['Photothrombotic stroke'],
        ['Microsphere injection']
    ]


@schema
class Surgery(dj.Manual):
    definition = """ # Table to keep track of surgeries on mice
    -> Mouse
    surgery_num         : tinyint        # Surgery number for this animal, start counting from 1
    ---
    surgery_date        : datetime       # Date and time of intervention (YYYY-MM-DD HH:MM:SS)
    -> SurgeryType 
    anesthesia          : varchar(2048)  # Type and dose of anesthesia used (e.g. 2 percent Isoflurane or Triple shot)
    pre_op_weight       : decimal(3,1)   # Pre-op weight in grams
    stroke_params       : varchar(2048)  # Stroke params such as illumination time, if applicable
    duration            : smallint       # Approximate duration of intervention, in minutes
    surgery_notes       : varchar(2048)  # Additional notes
    """

    def insert(self, rows, **kwargs):
        """Extend the insert method so that pre_op weights are automatically entered in Weight() table
        At the moment, only works if the rows are entered as dictionary data types
        This also covers usage of the insert1() function
        Matteo 2021-05-18
        ----------------------------------------------
        :param rows:  An iterable where an element is a numpy record, a dict-like object, a pandas.DataFrame, a sequence,
            or a query expression with the same heading as table self
        """
        # insert rows one by one
        for row in rows:
            # check if row is a dictionary
            if not isinstance(row, dict):
                # skip if row is not a dictionary data type
                # TODO: add handling of other datatypes
                print("Row is not a dictionary. It will be inserted without tracking the weight.")
                super().insert((row,), **kwargs)
                continue

            # get relevant info from row
            experimenter = row["username"]
            mouse_id = row["mouse_id"]
            date = row["surgery_date"]
            weight = row["pre_op_weight"]

            # get rid of hours and minutes by parsing date into datetime object, then back into string
            date_parsed = parse(date)
            date_str = date_parsed.strftime("%Y-%m-%d")

            # The "transaction" context ensures that the database is only changed if both inserts worked
            connection = Weight.connection
            with connection.transaction:
                if weight != 0:
                    #insert row into Weight table
                    Weight().insert1({"username": experimenter, "mouse_id": mouse_id, "date_of_weight": date_str,
                                      "weight": weight})

                # add row to Surgery table
                super().insert((row,), **kwargs)


@schema
class Injection(dj.Manual):
    definition = """ # Injections performed during surgery
    -> Surgery
    injection_num       : tinyint        # Injection number for this surgery, start counting from 1
    ---
    -> Substance                         # Link to substance lookup table
    volume              : float          # Injected volume in microliters
    dilution            : varchar(128)   # Dilution or concentration of the substance
    site                : varchar(128)   # Site of injection (Stereotaxic brain region, CCA, i.p. etc.)
    coordinates         : varchar(128)   # Stereotaxic coordinates of intervention, if applicable
    injection_notes     : varchar(2048)  # Additional notes
    """

