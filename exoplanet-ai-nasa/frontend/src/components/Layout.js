import React from 'react';
import { NavLink, Outlet } from 'react-router-dom';
import { 
  Telescope, 
  Database, 
  Brain, 
  BarChart3, 
  Settings, 
  Activity,
  Satellite,
  Globe2,
  Orbit
} from 'lucide-react';

const Layout = () => {
  const navItems = [
    { path: '/', icon: Activity, label: 'Dashboard', description: 'System Overview' },
    { path: '/data', icon: Database, label: 'Data Loading', description: 'NASA Data Sources' },
    { path: '/analysis', icon: BarChart3, label: 'Data Analysis', description: 'Explore Datasets' },
    { path: '/ml-training', icon: Brain, label: 'ML Training', description: 'Train Models' },
    { path: '/predictions', icon: Telescope, label: 'Predictions', description: 'Classify Exoplanets' },
    { path: '/visualizations', icon: Globe2, label: 'Visualizations', description: 'Charts & Plots' },
    { path: '/light-curves', icon: Orbit, label: 'Light Curves', description: 'Transit Analysis' }
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 shadow-sm">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Satellite className="h-8 w-8 text-blue-600" />
              <div>
                <h1 className="text-xl font-bold text-gray-900">Celestia</h1>
                <p className="text-sm text-gray-600">Exoplanet Classification & Research</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2 text-sm text-gray-600">
                <div className="h-2 w-2 rounded-full bg-green-400"></div>
                <span>System Operational</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      <div className="flex">
        {/* Sidebar Navigation */}
        <nav className="w-64 bg-white border-r border-gray-200 min-h-screen">
          <div className="p-4">
            <div className="space-y-1">
              {navItems.map((item) => {
                const Icon = item.icon;
                return (
                  <NavLink
                    key={item.path}
                    to={item.path}
                    className={({ isActive }) =>
                      `flex items-center px-3 py-3 text-sm rounded-lg transition-colors duration-150 ${
                        isActive
                          ? 'bg-blue-50 text-blue-700 border border-blue-200'
                          : 'text-gray-700 hover:bg-gray-100'
                      }`
                    }
                  >
                    <Icon className="h-5 w-5 mr-3 flex-shrink-0" />
                    <div className="flex-1 min-w-0">
                      <div className="font-medium">{item.label}</div>
                      <div className="text-xs text-gray-500 truncate">{item.description}</div>
                    </div>
                  </NavLink>
                );
              })}
            </div>

            {/* Quick Stats */}
            <div className="mt-8 p-4 bg-gray-50 rounded-lg">
              <h3 className="text-sm font-medium text-gray-900 mb-3">Quick Stats</h3>
              <div className="space-y-2 text-xs text-gray-600">
                <div className="flex justify-between">
                  <span>Data Records:</span>
                  <span className="font-medium" id="data-count">0</span>
                </div>
                <div className="flex justify-between">
                  <span>Trained Models:</span>
                  <span className="font-medium" id="model-count">0</span>
                </div>
                <div className="flex justify-between">
                  <span>Planet Types:</span>
                  <span className="font-medium">10</span>
                </div>
              </div>
            </div>

            {/* Mission Info */}
            <div className="mt-4 p-4 bg-blue-50 rounded-lg border border-blue-200">
              <h3 className="text-sm font-medium text-blue-900 mb-2">Data Sources</h3>
              <div className="space-y-1 text-xs text-blue-700">
                <div>• NASA Exoplanet Archive</div>
                <div>• Kepler Mission Data</div>
                <div>• TESS Observations</div>
                <div>• Transit Analysis</div>
              </div>
            </div>
          </div>
        </nav>

        {/* Main Content */}
        <main className="flex-1 p-6">
          <Outlet />
        </main>
      </div>
    </div>
  );
};

export default Layout;