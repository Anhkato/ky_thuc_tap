<x-app-layout>
    <x-slot name="header">
        <h2 class="font-semibold text-xl text-gray-800 leading-tight">
            {{ __('Dashboard') }}
        </h2>
    </x-slot>

    <div class="py-12">
        <div class="max-w-7xl mx-auto sm:px-6 lg:px-8">
            <div class="bg-white overflow-hidden shadow-sm sm:rounded-lg">
                <div class="p-6 text-gray-900">
                    {{ __("You're logged in!") }}
                </div>
            </div>
        </div>
    </div>
    <div class="py-12">
        <div class="max-w-7xl mx-auto sm:px-6 lg:px-8">
            <div class="bg-white overflow-hidden shadow-sm sm:rounded-lg">
                <div class="p-6 text-gray-900">
                <h2>Bảo mật 2 lớp (2FA)</h2>
                    @csrf
                    @if (! auth()->user()->two_factor_secret)
                    <form method="POST" action="{{ url('/user/two-factor-authentication') }}">
                        @csrf
                        <button type="submit">Bật xác thực 2 bước</button>
                    </form>
                    @else
                    <p>2FA đã bật</p>

                    <div>{!! auth()->user()->twoFactorQrCodeSvg() !!}</div>

                    <p>Mã khôi phục:</p>
                    <ul>
                        @foreach (json_decode(decrypt(auth()->user()->two_factor_recovery_codes), true) as $code)
                            <li>{{ $code }}</li>
                        @endforeach
                    </ul>

                    <form method="POST" action="{{ url('/user/two-factor-authentication') }}">
                        @csrf
                        @method('DELETE')
                        <button type="submit" class="btn btn-success">Tắt xác thực 2 bước</button>
                    </form>
                    @endif
                </div>
            </div>
        </div>

    </div>
    
</x-app-layout>


