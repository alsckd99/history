import { useEffect, useRef, useState, useCallback } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import Fuse from 'fuse.js';

// Leaflet 기본 마커 아이콘 수정 (CDN 사용)
const DefaultIcon = L.icon({
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
});
L.Marker.prototype.options.icon = DefaultIcon;

interface MapViewProps {
  // 현재 표시할 지명들 (키워드 JSON에서 추출된 지명)
  placeNames: string[];
  // 현재 시간 (영상 시간)
  currentTime: number;
}

interface PlaceData {
  name: string;
  chn_name?: string;
  modern_loc?: string;
  type?: string;
  lat: number;
  lon: number;
  source: 'old' | 'new';
  fullAddr?: string;
}

interface OldPlaceFeature {
  type: string;
  geometry: {
    type: string;
    coordinates: [number, number];
  };
  properties: {
    name: string;
    chn_name?: string;
    modern_loc?: string;
    type?: string;
  };
}

export default function MapView({ placeNames, currentTime }: MapViewProps) {
  const mapRef = useRef<L.Map | null>(null);
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const fuseRef = useRef<Fuse<OldPlaceFeature> | null>(null);
  const oldDataRef = useRef<{ features: OldPlaceFeature[] } | null>(null);
  const markersRef = useRef<L.LayerGroup | null>(null);  // 팝업 마커용
  const blueDotsRef = useRef<L.LayerGroup | null>(null); // 파란점 레이어 (유지됨)
  const displayedPlacesRef = useRef<Set<string>>(new Set());
  
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [currentPlace, setCurrentPlace] = useState<PlaceData | null>(null);

  // 지도 초기화
  useEffect(() => {
    if (!mapContainerRef.current || mapRef.current) return;

    // Leaflet 지도 생성
    const map = L.map(mapContainerRef.current, {
      preferCanvas: true,
      zoomControl: true,
    }).setView([36.5, 127.5], 7);

    // OpenStreetMap 타일
    L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '© OpenStreetMap',
      maxZoom: 19,
    }).addTo(map);

    // 마커 레이어 그룹
    markersRef.current = L.layerGroup().addTo(map);
    // 파란점 레이어 그룹 (키워드 등장 시에만 추가, 유지됨)
    blueDotsRef.current = L.layerGroup().addTo(map);
    mapRef.current = map;

    // 옛 지명 데이터 로드
    loadOldPlaceData();

    return () => {
      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
      }
    };
  }, []);

  // 옛 지명 데이터 로드 및 Fuse.js 초기화
  const loadOldPlaceData = async () => {
    try {
      setIsLoading(true);
      
      // map_module/data 폴더의 JSON 파일 로드
      // 프론트엔드에서 접근하려면 public 폴더에 복사하거나 API로 제공해야 함
      const response = await fetch('/oldmap_places_popup.json');
      
      // Content-Type 확인 (HTML이 반환되면 파일이 없는 것)
      const contentType = response.headers.get('content-type');
      if (!response.ok || !contentType?.includes('application/json')) {
        // 파일이 없으면 빈 데이터로 시작 (에러 아님)
        console.warn('옛 지명 데이터를 찾을 수 없습니다. 현대 지명 검색만 사용합니다.');
        oldDataRef.current = { features: [] };
        setIsLoading(false);
        return;
      }
      
      const data = await response.json();
      oldDataRef.current = data;

      // 초기에는 지도에 점을 표시하지 않음 (키워드 등장 시에만 표시)

      // Fuse.js 검색 엔진 초기화
      fuseRef.current = new Fuse(data.features, {
        keys: ['properties.name', 'properties.modern_loc', 'properties.chn_name'],
        threshold: 0.2,
      });

      setIsLoading(false);
    } catch (err) {
      // 파일이 없는 경우도 정상 동작 (현대 지명 검색만 사용)
      console.warn('옛 지명 데이터 없음, 현대 지명 검색만 사용:', err);
      oldDataRef.current = { features: [] };
      setIsLoading(false);
    }
  };

  // 지명 검색 (옛 지명 → 현대 API 순서)
  const searchPlace = useCallback(async (placeName: string): Promise<PlaceData | null> => {
    // 1. 옛 지명 검색 (Fuse.js)
    if (fuseRef.current) {
      const results = fuseRef.current.search(placeName);
      if (results.length > 0) {
        const best = results[0].item;
        return {
          name: best.properties.name,
          chn_name: best.properties.chn_name,
          modern_loc: best.properties.modern_loc,
          type: best.properties.type,
          lat: best.geometry.coordinates[1],
          lon: best.geometry.coordinates[0],
          source: 'old',
        };
      }
    }

    // 2. 현대 지명 검색 (Nominatim API)
    try {
      const response = await fetch(
        `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(placeName + ' 한국')}&limit=1`
      );
      if (response.ok) {
        const json = await response.json();
        if (json.length > 0) {
          const best = json[0];
          return {
            name: placeName,
            lat: parseFloat(best.lat),
            lon: parseFloat(best.lon),
            fullAddr: best.display_name,
            source: 'new',
          };
        }
      }
    } catch (err) {
      console.warn('현대 지명 검색 오류:', err);
    }

    return null;
  }, []);

  // 지도 이동 및 마커 표시
  const showPlaceOnMap = useCallback((place: PlaceData) => {
    if (!mapRef.current || !markersRef.current || !blueDotsRef.current) return;

    // 지도 이동 (줌 레벨 14 = 더 가깝게)
    mapRef.current.flyTo([place.lat, place.lon], 14, { duration: 1 });

    // 파란점 추가 (영구적으로 유지됨)
    const blueDot = L.circleMarker([place.lat, place.lon] as [number, number], {
      radius: 8,
      color: '#0077ff',
      fillColor: '#0077ff',
      fillOpacity: 0.6,
    });
    
    // 파란점에도 간단한 팝업 추가 (클릭 시)
    const dotPopup = `<strong>${place.name}</strong>${place.chn_name ? ` (${place.chn_name})` : ''}`;
    blueDot.bindPopup(dotPopup);
    blueDot.addTo(blueDotsRef.current);

    // 마커 추가 (현재 위치 표시용)
    const marker = L.marker([place.lat, place.lon]);
    
    // 팝업 내용
    let popupContent = '';
    if (place.source === 'old') {
      popupContent = `
        <div style="min-width: 180px;">
          <div style="background: #0077ff; color: white; padding: 8px 12px; margin: -10px -10px 10px -10px; border-radius: 4px 4px 0 0;">
            <strong>${place.name}</strong>
            ${place.chn_name ? `<span style="opacity: 0.8; font-size: 12px;"> (${place.chn_name})</span>` : ''}
          </div>
          <div style="font-size: 13px;">
            ${place.modern_loc ? `<div><strong>현재 위치:</strong> ${place.modern_loc}</div>` : ''}
            ${place.type ? `<div><strong>유형:</strong> ${place.type}</div>` : ''}
          </div>
        </div>
      `;
    } else {
      popupContent = `
        <div style="min-width: 180px;">
          <div style="background: #10b981; color: white; padding: 8px 12px; margin: -10px -10px 10px -10px; border-radius: 4px 4px 0 0;">
            <strong>${place.name}</strong>
          </div>
          <div style="font-size: 13px;">
            ${place.fullAddr ? `<div>${place.fullAddr}</div>` : ''}
          </div>
        </div>
      `;
    }

    marker.bindPopup(popupContent).addTo(markersRef.current!);
    
    // 팝업 자동 열기
    setTimeout(() => {
      marker.openPopup();
    }, 1600);

    setCurrentPlace(place);
  }, []);

  // placeNames가 변경되면 새 지명 검색 및 표시
  useEffect(() => {
    if (isLoading || placeNames.length === 0) return;

    const processNewPlaces = async () => {
      for (const placeName of placeNames) {
        // 이미 표시된 지명은 스킵
        if (displayedPlacesRef.current.has(placeName)) continue;

        // 지명 검색
        const placeData = await searchPlace(placeName);
        
        if (placeData) {
          displayedPlacesRef.current.add(placeName);
          showPlaceOnMap(placeData);
          
          // 다음 지명 처리 전 대기 (지도 이동 1.5초 + 팝업 표시 + 여유 시간)
          await new Promise(resolve => setTimeout(resolve, 3000));
        }
      }
    };

    processNewPlaces();
  }, [placeNames, isLoading, searchPlace, showPlaceOnMap]);

  // 영상 되감기 시 마커 초기화 (파란점은 유지)
  useEffect(() => {
    if (currentTime < 5 && displayedPlacesRef.current.size > 0) {
      // 영상이 처음으로 돌아가면 팝업 마커만 초기화 (파란점은 유지)
      if (markersRef.current) {
        markersRef.current.clearLayers();
      }
      // 파란점과 표시된 지명 기록은 유지
      // displayedPlacesRef.current.clear(); // 주석 처리 - 파란점 유지를 위해
      // blueDotsRef.current?.clearLayers(); // 파란점은 초기화하지 않음
      setCurrentPlace(null);
      
      // 지도 초기 위치로
      if (mapRef.current) {
        mapRef.current.flyTo([36.5, 127.5], 7, { duration: 1 });
      }
    }
  }, [currentTime]);

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      {/* 지도 컨테이너 */}
      <div 
        ref={mapContainerRef} 
        style={{ 
          width: '100%', 
          height: '100%',
          borderRadius: '8px',
          overflow: 'hidden',
        }} 
      />

      {/* 로딩 표시 */}
      {isLoading && (
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          background: 'rgba(0,0,0,0.7)',
          color: 'white',
          padding: '12px 20px',
          borderRadius: '8px',
          fontSize: '14px',
          zIndex: 1000,
        }}>
          지도 데이터 로딩 중...
        </div>
      )}

      {/* 에러 표시 */}
      {error && (
        <div style={{
          position: 'absolute',
          top: '10px',
          left: '10px',
          background: 'rgba(239, 68, 68, 0.9)',
          color: 'white',
          padding: '8px 12px',
          borderRadius: '6px',
          fontSize: '12px',
          zIndex: 1000,
        }}>
          {error}
        </div>
      )}

      {/* 현재 표시 중인 지명 */}
      {/* {currentPlace && (
        <div style={{
          position: 'absolute',
          bottom: '10px',
          left: '10px',
          background: currentPlace.source === 'old' ? 'rgba(0, 119, 255, 0.9)' : 'rgba(16, 185, 129, 0.9)',
          color: 'white',
          padding: '8px 14px',
          borderRadius: '6px',
          fontSize: '13px',
          zIndex: 1000,
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
        }}>
          <span>
            <strong>{currentPlace.name}</strong>
            {currentPlace.chn_name && <span style={{ opacity: 0.8 }}> ({currentPlace.chn_name})</span>}
          </span>
        </div>
      )} */}
    </div>
  );
}

