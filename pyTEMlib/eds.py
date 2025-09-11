for i, peak in reversed(list(enumerate(peaks))):
        for z in range(5, 82):
            edge_info  = pyTEMlib.eels_tools.get_x_sections(z)
            lines = edge_info.get('lines', {})
            if abs(lines.get('K-L3', {}).get('position', -50) - energy_scale[peak]) <40:
                elements.add(edge_info['name'])
                print(edge_info['name'])
                for key, line in lines.items():
                    dist = np.abs(energy_scale[peaks]-line.get('position', 0))
                    if key[0] == 'K' and np.min(dist)< 40:
                        ind = np.argmin(dist)
                        accounted_peaks.add(ind)
                        print(key, line.get('position'), ind , energy_scale[peaks][ind])
            elif abs(lines.get('K-L2', {}).get('position', -50) - energy_scale[peak]) <40:
                if abs(edge_info['lines']['K-L2']['position']- energy_scale[peak]) <30:
                    accounted_peaks.add(i)
                    elements.add(edge_info['name'])

            if abs(lines.get('L3-M5', {}).get('position', -50) - energy_scale[peak]) <40:
                elements.add(edge_info['name'])
                print(edge_info['name'])
                for key, line in lines.items():
                    dist = np.abs(energy_scale[peaks]-line.get('position', 0))
                    if key[0] == 'L' and np.min(dist)< 40:
                        ind = np.argmin(dist)
                        accounted_peaks.add(ind)
    return list(elements)