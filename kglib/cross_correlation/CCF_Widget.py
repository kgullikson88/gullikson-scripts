"""
Bokeh widget for analyzing CCF data.
"""

from collections import OrderedDict
import logging

from bokeh.models import ColumnDataSource, Plot, HoverTool
from bokeh.plotting import figure, curdoc
from bokeh.properties import String, Instance
from bokeh.server.app import bokeh_app
from bokeh.server.utils.plugins import object_page
from bokeh.models.widgets import HBox, VBox, VBoxForm, Select
from kglib.utils.HDF5_Helpers import Full_CCF_Interface, Kurucz_CCF_Interface, Primary_CCF_Interface


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Parse command-line arguments 
ADDMODE = 'simple'

class CCF_App(VBox):
    extra_generated_classes = [["CCF_App", "CCF_App", "VBox"]]
    jsmodel = "VBox"

    # data sources
    main_source = Instance(ColumnDataSource)
    current_source = Instance(ColumnDataSource)

    # layout boxes
    upper_row = Instance(HBox)  # Shows CCF height vs. temperature
    lower_row = Instance(HBox)   # Shows CCF

    # plots
    mainplot = Instance(Plot)
    ccf_plot = Instance(Plot)
    par_plot = Instance(Plot)

    # inputs
    star = String(default=u"HIP 92855")
    inst_date = String(default=u"CHIRON/20141015")
    star_select = Instance(Select)
    inst_date_select = Instance(Select)
    input_box = Instance(VBoxForm)

    #_ccf_interface = Kurucz_CCF_Interface(cache=False, update_cache=False)
    #_ccf_interface = Primary_CCF_Interface(cache=False, update_cache=False)
    _ccf_interface = Full_CCF_Interface(cache=False, update_cache=False)
    #                                    cache_fname='/home/kgullikson/.PythonModules/CCF_metadata.csv')
    _df_cache = {}


    def __init__(self, *args, **kwargs):
        super(CCF_App, self).__init__(*args, **kwargs)


    @classmethod
    def create(cls):
        """
        This function is called once, and is responsible for
        creating all objects (plots, datasources, etc)
        """
        # create layout widgets
        logging.info('Creating CCF_App')
        obj = cls()
        obj.upper_row = HBox()
        obj.lower_row = HBox()
        obj.input_box = VBoxForm()

        # create input widgets
        obj.set_defaults()
        obj.make_star_input()
        obj.make_inst_date_input()

        # outputs
        obj.make_source()
        obj.make_plots()

        # layout
        obj.set_children()
        return obj

    def set_defaults(self):
        stars = self._ccf_interface.list_stars()
        self.star = stars[0]
        dates = self._ccf_interface.get_observations(self.star)
        self.inst_date = '/'.join(dates[0])
        #self.star = 'HIP 79199'
        #self.inst_date = 'CHIRON/2014-03-18'


    def make_star_input(self):
        starnames = sorted(self._ccf_interface.list_stars())
        self.star_select = Select(
            name='Star identifier',
            value=self.star,
            options=starnames
        )

    def make_inst_date_input(self):
        observations = self._ccf_interface.get_observations(self.star)
        observations = ['/'.join(obs).ljust(20, ' ') for obs in observations]
        self.inst_date = observations[0]
        if isinstance(self.inst_date_select, Select):
            self.inst_date_select.update(value=observations[0], options=observations)
        else:
            self.inst_date_select = Select.create(
                name='Instrument/Date',
                value=observations[0],
                options=observations,
            )

    def make_source(self):
        self.main_source = ColumnDataSource(data=self.df)


    def plot_ccf(self, name, T, x_range=None):
        # Load the ccf from the HDF5 file.
        logging.debug('Plotting ccf name {}'.format(name))
        observation = self.inst_date
        i = observation.find('/')
        instrument = observation[:i]
        vel, corr = self._ccf_interface.load_ccf(instrument, name)

        # Now, plot
        p = figure(
            title='{} K'.format(T),
            x_range=x_range,
            plot_width=600, plot_height=400,
            title_text_font_size="10pt",
            tools="pan,wheel_zoom,box_select,reset,save"
        )
        p.line(vel, corr, line_width=2)
        p.xaxis[0].axis_label = 'Velocity (km/s)'
        p.yaxis[0].axis_label = 'CCF Power'

        return p

    def plot_Trun(self):
        star = self.star
        inst_date = self.inst_date

        data = self.selected_df
        idx = data.groupby(['T']).apply(lambda x: x['ccf_max'].idxmax())
        highest = data.ix[idx].copy()
        source = ColumnDataSource(data=highest)
        self.current_source = source

        p = figure(
            title="{} - {}".format(star, inst_date),
            plot_width=800, plot_height=400,
            tools="pan,wheel_zoom,tap,hover,reset",
            title_text_font_size="20pt",
        )
        p.circle("T", "ccf_max",
                 size=10,
                 nonselection_alpha=0.6,
                 source=source
        )
        p.xaxis[0].axis_label = 'Temperature (K)'
        p.yaxis[0].axis_label = 'CCF Peak Value'

        hover = p.select(dict(type=HoverTool))
        hover.tooltips = OrderedDict([
            ("Temperature", "@T"),
            ("vsini", "@vsini"),
            ("[Fe/H]", "@feh"),
            ("log(g)", "@logg"),
            ("Radial Velocity (km/s)", "@vel_max"),
            ("ccf peak height", "@ccf_max"),
        ])
        return p, highest


    def make_parplot(self):
        p = figure(
            title="CCF Parameters",
            plot_width=500, plot_height=400,
            tools="pan,wheel_zoom,box_select,reset",
            title_text_font_size="20pt",
        )
        p.circle("vsini", "feh",
                 size=12,
                 nonselection_alpha=0.003,
                 source=self.main_source
        )
        p.xaxis[0].axis_label = 'vsini (km/s)'
        p.yaxis[0].axis_label = '[Fe/H]'
        return p


    def make_plots(self):
        # Make the main plot (temperature vs ccf max value)
        self.mainplot, highest = self.plot_Trun()

        # Make the parameter plot (vsini vs. [Fe/H])
        self.par_plot = self.make_parplot()

        # Finally, make the CCF plot
        name, T = highest.sort('ccf_max', ascending=False)[['name', 'T']].values[0]
        self.ccf_plot = self.plot_ccf(name, T)
        return


    def set_children(self):
        self.children = [self.upper_row, self.lower_row]
        self.upper_row.children = [self.input_box, self.mainplot]
        self.input_box.children = [self.star_select, self.inst_date_select]
        self.lower_row.children = [self.ccf_plot, self.par_plot]

    def star_change(self, obj, attrname, old, new):
        logging.debug('Star change!')
        self.star = new
        self.make_inst_date_input()
        self.make_source()
        self.make_plots()
        self.set_children()
        curdoc().add(self)

    def inst_date_change(self, obj, attrname, old, new):
        logging.debug('Date change!')
        self.inst_date = new
        self.make_source()
        self.make_plots()
        self.set_children()
        curdoc().add(self)

    def setup_events(self):
        super(CCF_App, self).setup_events()
        if self.current_source:
            self.current_source.on_change('selected', self, 'Trun_change')
        if self.main_source:
            self.main_source.on_change('selected', self, 'par_change')
        if self.star_select:
            self.star_select.on_change('value', self, 'star_change')
        if self.inst_date_select:
            self.inst_date_select.on_change('value', self, 'inst_date_change')


    def Trun_change(self, obj, attrname, old, new):

        idx = int(new['1d']['indices'][0])
        T = self.current_source.data['T'][idx]
        name = self.current_source.data['name'][idx]
        logging.debug('T = {}\nName = {}\n'.format(T, name))
        self.ccf_plot = self.plot_ccf(name, T)
        self.set_children()
        curdoc().add(self)

    def par_change(self, obj, attrname, old, new):
        # Update plots
        self.mainplot, highest = self.plot_Trun()
        name, T = highest.sort('ccf_max', ascending=False)[['name', 'T']].values[0]
        self.ccf_plot = self.plot_ccf(name, T)

        self.set_children()
        curdoc().add(self)


    @property
    def df(self):
        # Parse the observation into an instrument and date
        observation = self.inst_date
        i = observation.find('/')
        instrument = observation[:i]
        date = observation[i+1:].strip()

        # Get the CCF summary
        starname = self.star 

        # Check if this setup has been cached
        key = (starname, instrument, date)
        if key in self._df_cache:
            return self._df_cache[key]

        df = self._ccf_interface.make_summary_df(instrument, starname, date, addmode=ADDMODE)
        df = df.rename(columns={'[Fe/H]': 'feh'})
        #self._df_cache[key] = df.copy()

        return df

    @property 
    def selected_df(self):
        df = self.df
        selected = self.main_source.selected['1d']['indices']
        if selected:
            df = df.iloc[selected, :]
        return df 

# The following code adds a "/bokeh/ccf/" url to the bokeh-server. This URL
# will render this CCF_App. If you don't want serve this applet from a Bokeh
# server (for instance if you are embedding in a separate Flask application),
# then just remove this block of code.
@bokeh_app.route("/bokeh/ccf/")
@object_page("ccf")
def make_ccf_app():
    app = CCF_App.create()
    return app
