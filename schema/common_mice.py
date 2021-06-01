"""Schema for mouse related information"""

import datajoint as dj
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
        ['Thy1-GCaMP6f', 'GP5.17', 'GCaMP6f', 'L2/3, L5, CA1, CA3, dentate gyrus'],
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
    #Todo: make cage number updateable, maybe in mouse GUI?

@schema
class Weight(dj.Manual):
    definition = """ # Table that stores the weights of individual mice
    -> Mouse
    date_of_weight      : date           # Date of weighing (year-month-day)
    ---
    weight              : decimal(3,1)   # Weight in grams
    """


@schema
class Sacrificed(dj.Manual):
    definition = """ # Table to keep track of euthanized mice
    -> Mouse
    ---
    date_of_sacrifice   : date           # Date of sacrifice (year-month-day)
    perfused            : tinyint        # 0 for no, 1 for yes (brain fixed and kept)
    reason              : varchar(1024)  # Comments
    """

    # example content (added with insert statement)
    # contents = [
    #     ['Brit', '2019-06-04', 'Window was not clear anymore'],
    # ]


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
       ['AAV9-hSyn-GCaMP6f', 'pAAV.Syn.GCaMP6f.WPRE.SV40', 'virus', 'addgene', 'https://www.addgene.org/100837/'],
       ['endothelin', 'endothelin 1 (ET-1)', 'vasoconstrictor', 'Sigma', 'https://www.sigmaaldrich.com/catalog/product/sigma/e7764'],
       ['microspheres', 'Fluorescent PMMA Microparticles Red5 20 m', 'microspheres', 'PolyAn',
         'https://www.poly-an.de/micro-nanoparticles/fluorescent-pmma-microparticles/fluorescent-pmma-microparticles'],
       ['TMP', "Trimethoprim", "antibiotic", "Sigma", "https://www.sigmaaldrich.com/catalog/product/sigma/t7883"],
    ]


@schema
class Surgery(dj.Manual):
    definition = """ # Table to keep track of surgeries on mice
    -> Mouse
    surgery_num         : tinyint        # Surgery number for this animal, start counting from 1
    ---
    surgery_date        : datetime       # Date and time of intervention (YYYY-MM-DD HH:MM:SS)
    surgery_type        : varchar(2048)  # Description of surgery (e.g. headmount implantation)
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

            # check if row is already present in Weight table
            if len(Weight() & "username='{}'".format(experimenter) & "mouse_id = '{}'".format(mouse_id) & "date_of_weight = '{}'".format(date)) > 0:
                print("A weight has already been recorded for this mouse and date, pre_op_weight will not be added to Weights table.")
            else:
                #insert row into Weight table
                Weight().insert1({"username":experimenter, "mouse_id":mouse_id, "date_of_weight":date, "weight":weight})

            # add row to Surgery table
            super().insert((row,), **kwargs)


@schema
class Injection(dj.Manual):
    definition = """ # Holds injection data for each surgery
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
