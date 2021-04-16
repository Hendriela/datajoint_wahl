"""Schema for mouse related information"""

import datajoint as dj
schema = dj.schema('common_mice', locals(), create_tables=True)


@schema
class Investigator(dj.Lookup):
    definition = """    # Name and contact information of lab members
    username        : varchar(128)    # Unique username, should be the UZH shortname
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
    strain          : char(128)       # Mouse variant short name
    ---
    formal_name     : varchar(2048)   # More detailled description of genetic background
    ca_indicator    : varchar(255)    # Expressed calcium indicator (e.g. GCaMP6f)
    layer           : varchar(255)    # Expression layer of indicator (e.g. L2/3 or all)
    """
    contents = [
        ['WT', 'C57BL/6J', 'None', 'None'],
        ['Snap25-RCaMP', 'Snap-Cre;tTA-C;RCaMP', 'RCaMP', 'all'],
        ['L2/3-RCaMP', 'L2/3Cre;tTA-C;RCaMP', 'RCaMP', 'L2/3'],
        ['L2/3-GCaMP6f', 'L2/3Cre;tTA-C;GCa6f ', 'GCaMP6f', 'L2/3'],
        ['Snap25-GCaMPf', 'Snap-Cre;tTA-C;GCaP6 ', 'GCaMP6f', 'all'],
    ]


@schema
class Licence(dj.Lookup):
    definition = """    # Licence and project ID under which the mouse is kept
    licence_id      : varchar(128)      # Licence ID with project suffix to keep it unique
    ---
    -> Investigator                     # Link to licence holder 
    description     : varchar(512)      # Short description of the project
    document        : attach@localstore # Path to the locally stored PDF document of the licence
    """
    # add contents manually to add licence file
    # contents = [
    #    ['241/2018-A', 'aswahl', 'Studying sensorimotor recovery after stroke', '???'],
    #    ['241/2018-B', 'aswahl', 'Studying cognitive impairment after stroke', '???']
    # ]


@schema
class Mouse(dj.Manual):
    definition = """ # Basic information about the Mouse
      -> Investigator                          # Link to investigator to which this mouse belongs
      mouse_id      : int                      # Name of mouse (unique per investigator, without prefix)
      ---
      dob           : date                     # Day of birth (year-month-day)
      sex           : enum('M', 'F', 'U')      # Sex of mouse - Male, Female, or Unknown/Unclassified
      batch         : int                      # Batch ID that this mouse belongs to (0 if no batch)
      -> Strain                                # Link to the genetic type of the mouse
      genotype      : varchar(128)              # Genotype
      irats_id      : varchar(20)              # ID that is used in iRats (e.g. BJ5698 RR)
      cage_num      : int                      # Cage number
      ear_mark      : varchar(10)              # Actual ear mark
      -> Licence                               # Link to project ID
      info          : varchar(1024)            # Additional information about this mouse
      """


@schema
class Weight(dj.Manual):
    definition = """ # Table that stores the weights of individual mice
    -> Mouse
    date_of_weight      : date           # Date of weighing (year-month-day)
    ---
    weight              : float          # Weight in grams
    """


@schema
class Sacrificed(dj.Manual):
    definition = """ # Table to keep track of euthanized mice
    -> Mouse
    ---
    date_of_sacrifice   : date           # Date of sacrifice (year-month-day)
    reason              : varchar(1024)  # Comments
    """

    # example content (added with insert statement)
    # contents = [
    #     ['Brit', '2019-06-04', 'Window was not clear anymore'],
    # ]

@schema
class Surgery(dj.Manual):
    definition = """ # Table to keep track of surgeries on mice
    -> Mouse
    surgery_num         : int            # Surgery ID
    ---
    surgery_date        : datetime       # Date of intervention (year-month-day)
    surgery_type        : varchar(2048)  # Description of surgery (e.g. "headmount implantation")
    coordinates         : varchar(2048)  # Stereotaxic coordinates of intervention, if applicable
    anesthesia          : varchar(2048)  # Type and dose of anesthesia used
    weight_grams        : float          # Pre-op weight in grams
    stroke_params       : varchar(2048)  # Stroke params such as illumination time, if applicable
    duration            : int            # Approximate duration of intervention, in minutes
    notes               : varchar(2048)  # Additional notes
    """

    # example content (added with insert statement)
    # contents = [
    #     ['Brit', '2019-06-04', 'Window was not clear anymore'],
    # ]
