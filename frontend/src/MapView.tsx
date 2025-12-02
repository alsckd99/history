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
  // 현재 시간 (영상 시간)
  currentTime: number;
  // 영상 ID (map_keyword.json 로드용)
  videoId?: string;
}

interface TimelineItem {
  time: number;
  type: 'old' | 'new';
  name: string;           // 검색용 이름 (예: "노량리")
  displayName?: string;   // 표시용 이름 (예: "노량해전") - 없으면 name 사용
}

interface PlaceData {
  name: string;           // 검색에 사용된 이름
  displayName?: string;   // 표시용 이름 (없으면 name 사용)
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

export default function MapView({ currentTime, videoId }: MapViewProps) {
  const mapRef = useRef<L.Map | null>(null);
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const fuseRef = useRef<Fuse<OldPlaceFeature> | null>(null);
  const oldDataRef = useRef<{ features: OldPlaceFeature[] } | null>(null);
  const markersRef = useRef<L.LayerGroup | null>(null);  // 팝업 마커용
  const blueDotsRef = useRef<L.LayerGroup | null>(null); // 파란점 레이어 (유지됨)
  
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [currentPlace, setCurrentPlace] = useState<PlaceData | null>(null);
  
  // 타임라인 데이터
  const [timelineData, setTimelineData] = useState<TimelineItem[]>([]);
  // 이미 표시한 시간 인덱스 추적
  const displayedTimesRef = useRef<Set<number>>(new Set());
  // 마지막 처리 시간
  const lastProcessedTimeRef = useRef<number>(-1);

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
    // 파란점 레이어 그룹
    blueDotsRef.current = L.layerGroup().addTo(map);
    mapRef.current = map;

    // 데이터 로드
    loadOldPlaceData();
    loadMapKeywordData();

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
      const response = await fetch(`${import.meta.env.BASE_URL}oldmap_places_popup.json`);
      
      const contentType = response.headers.get('content-type');
      if (!response.ok || !contentType?.includes('application/json')) {
        console.warn('옛 지명 데이터를 찾을 수 없습니다. 현대 지명 검색만 사용합니다.');
        oldDataRef.current = { features: [] };
        setIsLoading(false);
        return;
      }
      
      const data = await response.json();
      oldDataRef.current = data;

      // Fuse.js 검색 엔진 초기화
      fuseRef.current = new Fuse(data.features, {
        keys: ['properties.name', 'properties.modern_loc', 'properties.chn_name'],
        threshold: 0.2,
      });

      setIsLoading(false);
    } catch (err) {
      console.warn('옛 지명 데이터 없음, 현대 지명 검색만 사용:', err);
      oldDataRef.current = { features: [] };
      setIsLoading(false);
    }
  };

  // map_keyword.json 로드 (로컬 파일 우선)
  const loadMapKeywordData = async () => {
    try {
      // public 폴더에서 직접 로드 (GitHub Pages 호환)
      const localResponse = await fetch(`${import.meta.env.BASE_URL}map_keyword.json`);
      if (localResponse.ok) {
        const data = await localResponse.json();
        if (data.timelineData && Array.isArray(data.timelineData)) {
          setTimelineData(data.timelineData);
          console.log('[MapView] 타임라인 데이터 로드:', data.timelineData.length, '개');
          return;
        }
      }
      
      // 로컬 실패 시 API 서버 시도 (개발 환경용)
      const apiResponse = await fetch('http://localhost:8080/map-keywords');
      if (apiResponse.ok) {
        const data = await apiResponse.json();
        if (data.timelineData && Array.isArray(data.timelineData)) {
          setTimelineData(data.timelineData);
          console.log('[MapView] API 타임라인 데이터 로드:', data.timelineData.length, '개');
        }
      }
    } catch (err) {
      console.warn('[MapView] map_keyword.json 로드 실패:', err);
    }
  };

  // 지명 검색 (옛 지명 / 현대 API)
  const searchPlace = useCallback(async (placeName: string, searchType: 'old' | 'new', displayName?: string): Promise<PlaceData | null> => {
    console.log(`[MapView] 지명 검색: '${placeName}' (type: ${searchType})`);
    console.log(`[MapView] Fuse 초기화 여부: ${fuseRef.current ? '완료' : '미완료'}`);
    console.log(`[MapView] 옛 지명 데이터: ${oldDataRef.current?.features?.length || 0}개`);
    
    // old 타입이면 옛 지명 DB에서만 검색
    if (searchType === 'old') {
      if (!fuseRef.current) {
        console.warn(`[MapView] 옛 지명 DB가 로드되지 않음 - '${placeName}' 검색 불가`);
        return null;  // fallback 없이 null 반환
      }
      
      const results = fuseRef.current.search(placeName);
      console.log(`[MapView] 옛 지명 검색 결과: ${results.length}개`);
      
      if (results.length > 0) {
        const best = results[0].item;
        console.log(`[MapView] 옛 지명 발견: ${best.properties.name}`);
        return {
          name: best.properties.name,
          displayName: displayName || best.properties.name,
          chn_name: best.properties.chn_name,
          modern_loc: best.properties.modern_loc,
          type: best.properties.type,
          lat: best.geometry.coordinates[1],
          lon: best.geometry.coordinates[0],
          source: 'old',
        };
      }
      
      console.warn(`[MapView] 옛 지명 DB에서 '${placeName}' 찾을 수 없음`);
      return null;  // old 타입은 fallback 없이 null 반환
    }

    // new 타입만 현대 지명 검색 (Nominatim API)
    if (searchType === 'new') {
      try {
        const url = `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(placeName + ' 한국')}&limit=1`;
        console.log(`[MapView] Nominatim API 호출: ${url}`);
        
        const response = await fetch(url);
        console.log(`[MapView] Nominatim 응답 상태: ${response.status}`);
        
        if (response.ok) {
          const json = await response.json();
          console.log(`[MapView] Nominatim 결과: ${json.length}개`, json);
          
          if (json.length > 0) {
            const best = json[0];
            console.log(`[MapView] 현대 지명 발견: ${placeName} -> (${best.lat}, ${best.lon})`);
            return {
              name: placeName,
              displayName: displayName || placeName,  // 표시용 이름 사용
              lat: parseFloat(best.lat),
              lon: parseFloat(best.lon),
              fullAddr: best.display_name,
              source: 'new',
            };
          } else {
            console.warn(`[MapView] Nominatim에서 '${placeName}' 결과 없음`);
          }
        } else {
          console.warn(`[MapView] Nominatim API 오류: ${response.status}`);
        }
      } catch (err) {
        console.error('[MapView] 현대 지명 검색 오류:', err);
      }
    }

    return null;
  }, []);

  // 지도 이동 및 마커 표시
  const showPlaceOnMap = useCallback((place: PlaceData) => {
    if (!mapRef.current || !markersRef.current || !blueDotsRef.current) return;

    // 지도 이동
    mapRef.current.flyTo([place.lat, place.lon], 11, { duration: 1.5 });

    // 표시용 이름 (displayName이 있으면 사용, 없으면 name)
    const showName = place.displayName || place.name;

    // 파란점 추가 (영구적으로 유지됨) - 통일된 색상
    const blueDot = L.circleMarker([place.lat, place.lon] as [number, number], {
      radius: 8,
      color: '#0077ff',
      fillColor: '#0077ff',
      fillOpacity: 0.6,
    });
    
    const dotPopup = `<strong>${showName}</strong>${place.chn_name ? ` (${place.chn_name})` : ''}`;
    blueDot.bindPopup(dotPopup);
    blueDot.addTo(blueDotsRef.current);

    // 마커 추가 (현재 위치 표시용)
    markersRef.current.clearLayers(); // 이전 마커 제거
    const marker = L.marker([place.lat, place.lon]);
    
    // 팝업 내용 - 통일된 스타일
    const popupContent = `
      <div style="min-width: 180px;">
        <div style="background: #0077ff; color: white; padding: 8px 12px; margin: -10px -10px 10px -10px; border-radius: 4px 4px 0 0;">
          <strong>${showName}</strong>
          ${place.chn_name ? `<span style="opacity: 0.8; font-size: 12px;"> (${place.chn_name})</span>` : ''}
        </div>
        <div style="font-size: 13px;">
          ${place.modern_loc ? `<div><strong>현재 위치:</strong> ${place.modern_loc}</div>` : ''}
          ${place.type ? `<div><strong>유형:</strong> ${place.type}</div>` : ''}
          ${place.fullAddr ? `<div>${place.fullAddr}</div>` : ''}
        </div>
      </div>
    `;

    marker.bindPopup(popupContent).addTo(markersRef.current!);
    
    // 팝업 자동 열기
    setTimeout(() => {
      marker.openPopup();
    }, 1600);

    setCurrentPlace(place);
  }, []);

  // currentTime에 따라 타임라인 처리
  useEffect(() => {
    if (isLoading || timelineData.length === 0) return;

    // 현재 시간에 해당하는 이벤트 찾기
    const currentEvents = timelineData.filter(item => {
      // 해당 시간이 지났고, 아직 표시하지 않은 이벤트
      return item.time <= currentTime && !displayedTimesRef.current.has(item.time);
    });

    if (currentEvents.length > 0) {
      // 가장 최근 이벤트 처리
      const latestEvent = currentEvents[currentEvents.length - 1];
      
      console.log(`[MapView] 시간 ${currentTime}초 - '${latestEvent.name}' (${latestEvent.type}) 검색`);
      
      // 표시 완료로 마킹
      displayedTimesRef.current.add(latestEvent.time);
      
      // 지명 검색 및 표시 (displayName이 있으면 표시용으로 사용)
      searchPlace(latestEvent.name, latestEvent.type, latestEvent.displayName).then(placeData => {
        if (placeData) {
          showPlaceOnMap(placeData);
        } else {
          console.warn(`[MapView] '${latestEvent.name}' 검색 결과 없음`);
        }
      });
    }
  }, [currentTime, isLoading, timelineData, searchPlace, showPlaceOnMap]);

  // 영상 되감기 시 초기화
  useEffect(() => {
    if (currentTime < 2 && displayedTimesRef.current.size > 0) {
      console.log('[MapView] 영상 되감기 - 초기화');
      
      // 마커 초기화
      if (markersRef.current) {
        markersRef.current.clearLayers();
      }
      if (blueDotsRef.current) {
        blueDotsRef.current.clearLayers();
      }
      
      // 표시 기록 초기화
      displayedTimesRef.current.clear();
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
      {currentPlace && (
        <div style={{
          position: 'absolute',
          bottom: '10px',
          left: '10px',
          background: 'rgba(0, 119, 255, 0.9)',
          color: 'white',
          padding: '8px 14px',
          borderRadius: '6px',
          fontSize: '13px',
          zIndex: 1000,
        }}>
          <strong>{currentPlace.displayName || currentPlace.name}</strong>
          {currentPlace.chn_name && <span style={{ opacity: 0.8 }}> ({currentPlace.chn_name})</span>}
        </div>
      )}
    </div>
  );
}
