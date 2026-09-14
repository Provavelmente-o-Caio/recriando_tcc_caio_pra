using System;
using System.Collections.Generic;

using Slb.Ocean.Core;

namespace missing_data
{
    public class Plugin : Slb.Ocean.Core.Plugin
    {
        public override string AppVersion
        {
            get { return "2023.1"; }
        }

        public override string Author
        {
            get { return "Caio Ferreira Cardoso"; }
        }

        public override string Contact
        {
            get { return "caiofecardoso@hotmail.com"; }
        }

        public override IEnumerable<PluginIdentifier> Dependencies
        {
            get { return null; }
        }

        public override string Description
        {
            get { return ""; }
        }

        public override string ImageResourceName
        {
            get { return null; }
        }

        public override Uri PluginUri
        {
            get { return new Uri("https://www.pluginuri.info"); }
        }

        public override IEnumerable<ModuleReference> Modules
        {
            get 
            {
                yield return new ModuleReference(typeof(missing_data.Module)); 
                // Please fill this method with your modules with lines like this:
                //yield return new ModuleReference(typeof(Module));

            }
        }

        public override string Name
        {
            get { return "Plugin"; }
        }

        public override PluginIdentifier PluginId
        {
            get { return new PluginIdentifier(GetType().FullName, GetType().Assembly.GetName().Version); }
        }

        public override ModuleTrust Trust
        {
            get { return new ModuleTrust("Default"); }
        }
    }
}
