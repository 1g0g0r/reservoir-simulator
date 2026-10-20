// Клиентские колбэки визуализации (`visualisation.py`, ClientsideFunction('viz', ...)): работают без запросов к серверу.
window.dash_clientside = Object.assign({}, window.dash_clientside, {
    viz: {
        // Подпись времени над ползунком - живая и при перетаскивании
        timeLabel: function (dv, v, view, times) {
            if (!view || view.kind === 'series') return '';
            const idx = (dv !== null && dv !== undefined) ? dv : ((v !== null && v !== undefined) ? v : 0);
            const t = (times && times[idx] !== undefined) ? times[idx] : 0;
            return 'Время: ' + Number(t).toFixed(1) + ' день';
        },

        // Заслонка ползунка: не больше одного запроса слоя в работе, новейшее положение перекрывает старые.
        // Флаг снимает `running` у `_on_slider` (и при ошибке); сторожевой таймер - на случай, если ответ потерялся
        gate: function (dv, v, busy) {
            const S = window._live || (window._live = {want: null, busy: false, last: null, seq: 0, timer: null});
            const trig = dash_clientside.callback_context.triggered.map(t => t.prop_id);
            if (trig.includes('live-busy.data')) {
                S.busy = !!busy;
                if (!S.busy) clearTimeout(S.timer);
            }
            if (trig.includes('time-slider.value')) S.want = v;
            else if (trig.includes('time-slider.drag_value')) S.want = dv;
            if (S.busy || S.want === null || S.want === undefined || S.want === S.last) {
                return dash_clientside.no_update;
            }
            const t = S.want;
            S.want = null; S.last = t; S.busy = true;
            clearTimeout(S.timer);
            S.timer = setTimeout(() => dash_clientside.set_props('live-busy', {data: false}), 2000);
            return {t: t, seq: ++S.seq};   // seq - чтобы повтор того же t тоже считался изменением
        },
    },
});
